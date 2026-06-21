import datetime
import hmac
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np
import pandas as pd
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    callback_context,
    dash_table,
    dcc,
    html,
    no_update,
)
from dash.exceptions import PreventUpdate
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent
os.chdir(str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tools.maps import WC26_TEAMS

# ── Palette ───────────────────────────────────────────────────────────────────
TEAM_COLORS = {
    "US": "#002868",
    "MX": "#006847",
    "CA": "#D2001C",
    "CW": "#002395",
    "HT": "#00209F",
    "PA": "#DA121A",
    "AR": "#4A8CC3",
    "BR": "#009C3B",
    "CO": "#C8A000",
    "EC": "#003580",
    "PY": "#D52B1E",
    "UY": "#5EB6E4",
    "AT": "#ED2939",
    "BE": "#C8102E",
    "BA": "#002395",
    "HR": "#CC0000",
    "CZ": "#D7141A",
    "EN": "#CF101A",
    "FR": "#002395",
    "DE": "#3A3A3A",
    "NL": "#E07020",
    "NO": "#EF2B2D",
    "PT": "#006600",
    "SQ": "#003DA5",
    "ES": "#AA151B",
    "SE": "#006AA7",
    "CH": "#D52B1E",
    "TR": "#E30A17",
    "DZ": "#006233",
    "CV": "#003893",
    "CD": "#0070C0",
    "CI": "#D4690F",
    "EG": "#CE1126",
    "GH": "#006B3F",
    "MA": "#C1272D",
    "SN": "#00853F",
    "ZA": "#007A4D",
    "TN": "#E70013",
    "AU": "#00843D",
    "IQ": "#007A3D",
    "IR": "#239F40",
    "JP": "#BC002D",
    "JO": "#007A3D",
    "QA": "#8D1B3D",
    "SA": "#006C35",
    "KR": "#003478",
    "UZ": "#1EB53A",
    "NZ": "#404040",
}

BG = "#F9F7F4"
BG2 = "#F2F0EC"
BG3 = "#E9E6E1"
BORDER = "#DDD9D2"
TEXT = "#1C1917"
TDIM = "#78716C"
ACCENT = "#2563EB"

BARLOW = "'Barlow Condensed', sans-serif"
INTER = "'Inter', -apple-system, sans-serif"

ADMIN_PASSWORD = os.environ.get("WC_ADMIN_PASSWORD", "admin1234")

PIPELINE_STAGES = [
    ("📡  Scrape Elo", "elo.py", "eloratings.net → data/elo_results.csv"),
    ("📊  Process Stats", "stats.py", "footy_stats_data/ → data/fs_results.csv"),
    ("🔧  Build Features", "features.py", "elo + stats → data/features.csv"),
    ("🤖  Train Model", "model.py", "eval + deploy models → models/*.txt"),
    ("📈  Evaluate", "test.py", "compare models on WC2026 holdout"),
    ("🔮  Predict Upcoming", "fixtures.py", "live fixtures + deploy model"),
    (
        "💾  Save Predictions",
        "save_predictions.py",
        "snapshot today's upcoming preds → data/prediction_snapshots/",
    ),
    (
        "🕰️  Retro Predictions",
        "retro_predict.py",
        "day-by-day backfill for past WC2026 games → data/retro_predictions.csv",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _lum(h):
    c = h.lstrip("#")
    if len(c) == 3:
        c = c[0] * 2 + c[1] * 2 + c[2] * 2
    r, g, b = int(c[:2], 16), int(c[2:4], 16), int(c[4:], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _fg(bg):
    return "#000" if _lum(bg) > 0.45 else "#fff"


def full(code):
    return WC26_TEAMS.get(code, code)


def parse_date(gid):
    p = gid.split("_")
    return datetime.date(2000 + int(p[0]), int(p[1]), int(p[2]))


def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def top_scorelines(lam_t, lam_o, n=8, max_g=8):
    grid = np.outer(
        poisson.pmf(range(max_g + 1), lam_t), poisson.pmf(range(max_g + 1), lam_o)
    )
    rows = []
    for i in range(max_g + 1):
        for j in range(max_g + 1):
            result = "W" if i > j else ("D" if i == j else "L")
            rows.append(
                {"score": f"{i}–{j}", "prob": grid[i, j] * 100, "result": result}
            )
    return sorted(rows, key=lambda x: x["prob"], reverse=True)[:n]


def prob_to_american(prob_pct):
    p = prob_pct / 100
    if p <= 0 or p >= 1:
        return ""
    if p >= 0.5:
        return f"{int(round(-(p / (1 - p)) * 100))}"
    return f"+{int(round(((1 - p) / p) * 100))}"


def ou_probs(lam_total, line):
    k = int(line)
    if line == k:  # integer line — push possible
        p_over = 1 - poisson.cdf(k, lam_total)
        p_under = poisson.cdf(k - 1, lam_total)
        p_push = poisson.pmf(k, lam_total)
    else:  # .5 line — no push
        p_over = 1 - poisson.cdf(k, lam_total)
        p_under = poisson.cdf(k, lam_total)
        p_push = 0.0
    return p_over, p_under, p_push


# ── Data ──────────────────────────────────────────────────────────────────────
def load_preds():
    from fixtures import predict_upcoming

    return predict_upcoming()


def build_game_df(preds):
    seen, rows = set(), []
    for _, r in preds.iterrows():
        gid = r["game_id"]
        if gid in seen:
            continue
        seen.add(gid)
        d = parse_date(gid)
        rows.append(
            {
                "game_id": gid,
                "team": r["team"],
                "opponent": r["opponent"],
                "date_str": d.strftime("%b %d"),
                "date_iso": d.isoformat(),
                "exp_goals": float(r["exp_goals"]),
                "opp_exp_goals": float(r["opp_exp_goals"]),
                "p_win": float(r["p_win"]),
                "p_draw": float(r["p_draw"]),
                "p_loss": float(r["p_loss"]),
            }
        )
    return pd.DataFrame(rows)


# ── Components ────────────────────────────────────────────────────────────────


def _banner_panels(r, scale=1.0):
    tc, oc = TEAM_COLORS.get(r["team"], "#444"), TEAM_COLORS.get(r["opponent"], "#444")
    tfg, ofg = _fg(tc), _fg(oc)

    def panel(code, name, col, fg, xg, pct, label):
        fs_code = f"{3.4 * scale:.2f}rem"
        fs_pct = f"{2.6 * scale:.2f}rem"
        pad_v = f"{int(22 * scale)}px"
        pad_h = f"{int(32 * scale)}px"
        mb_xg = f"{int(12 * scale)}px"
        mb_name = f"{int(5 * scale)}px"
        pt_sep = f"{int(10 * scale)}px"
        return html.Div(
            [
                html.Div(
                    name,
                    style={
                        "fontFamily": BARLOW,
                        "fontSize": "0.72rem",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.16em",
                        "opacity": "0.7",
                        "marginBottom": mb_name,
                    },
                ),
                html.Div(
                    code,
                    style={
                        "fontFamily": BARLOW,
                        "fontSize": fs_code,
                        "fontWeight": "800",
                        "lineHeight": "1",
                        "marginBottom": "3px",
                    },
                ),
                html.Div(
                    f"xG {xg:.2f}",
                    style={
                        "fontSize": "0.82rem",
                        "opacity": "0.65",
                        "marginBottom": mb_xg,
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            f"{pct:.1f}%",
                            style={
                                "fontFamily": BARLOW,
                                "fontSize": fs_pct,
                                "fontWeight": "800",
                                "lineHeight": "1",
                            },
                        ),
                        html.Div(
                            label,
                            style={
                                "fontSize": "0.65rem",
                                "fontWeight": "700",
                                "textTransform": "uppercase",
                                "letterSpacing": "0.14em",
                                "opacity": "0.6",
                                "marginTop": "2px",
                            },
                        ),
                    ],
                    style={
                        "borderTop": "1px solid rgba(255,255,255,0.2)",
                        "paddingTop": pt_sep,
                    },
                ),
            ],
            style={
                "flex": "1",
                "background": col,
                "color": fg,
                "padding": f"{pad_v} {pad_h}",
                "textAlign": "center",
            },
        )

    draw_fs = f"{2.0 * scale:.2f}rem"
    center_p = f"{int(20 * scale)}px {int(22 * scale)}px"
    center_gap = f"{int(14 * scale)}px"
    center = html.Div(
        [
            html.Div(
                "VS",
                style={
                    "fontFamily": BARLOW,
                    "fontSize": "0.85rem",
                    "fontWeight": "800",
                    "letterSpacing": "0.16em",
                    "color": TDIM,
                },
            ),
            html.Div(
                [
                    html.Div(
                        f"{r['p_draw'] * 100:.1f}%",
                        style={
                            "fontFamily": BARLOW,
                            "fontSize": draw_fs,
                            "fontWeight": "800",
                            "color": TEXT,
                            "lineHeight": "1",
                        },
                    ),
                    html.Div(
                        "Draw",
                        style={
                            "fontSize": "0.6rem",
                            "fontWeight": "700",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.12em",
                            "color": TDIM,
                            "marginTop": "3px",
                        },
                    ),
                ]
            ),
        ],
        style={
            "background": BG2,
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": center_p,
            "gap": center_gap,
            "minWidth": f"{int(110 * scale)}px",
            "borderLeft": f"1px solid {BORDER}",
            "borderRight": f"1px solid {BORDER}",
        },
    )

    return (
        panel(
            r["team"], full(r["team"]), tc, tfg, r["exp_goals"], r["p_win"] * 100, "Win"
        ),
        center,
        panel(
            r["opponent"],
            full(r["opponent"]),
            oc,
            ofg,
            r["opp_exp_goals"],
            r["p_loss"] * 100,
            "Win",
        ),
    )


def match_card(r):
    left, center, right = _banner_panels(r, scale=0.68)
    return html.Div(
        [
            html.Div(
                [left, center, right],
                style={
                    "display": "flex",
                    "borderRadius": "12px",
                    "overflow": "hidden",
                    "boxShadow": "0 3px 20px rgba(0,0,0,0.18)",
                },
            ),
        ],
        id={"type": "match-banner", "index": r["game_id"]},
        n_clicks=0,
        className="match-card",
        style={"marginBottom": "10px", "cursor": "pointer"},
    )


def modal_match_header(r):
    left, center, right = _banner_panels(r, scale=0.75)
    return html.Div(
        [left, center, right],
        style={
            "display": "flex",
            "borderRadius": "12px",
            "overflow": "hidden",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.5)",
        },
    )


def scorelines_panel(lam_t, lam_o, tc, oc):
    rows = top_scorelines(lam_t, lam_o)
    top = rows[0]["prob"] if rows else 1
    rc_map = {"W": tc, "D": "#374151", "L": oc}

    def row(s):
        rc = rc_map.get(s["result"], "#374151")
        rfc = _fg(rc)
        bw = min(s["prob"] / top * 100, 100)
        american = prob_to_american(s["prob"])
        return html.Div(
            [
                html.Span(
                    s["score"],
                    style={
                        "fontFamily": BARLOW,
                        "fontSize": "1.4rem",
                        "fontWeight": "700",
                        "minWidth": "50px",
                        "fontVariantNumeric": "tabular-nums",
                        "color": TEXT,
                    },
                ),
                html.Span(
                    s["result"],
                    style={
                        "background": rc,
                        "color": rfc,
                        "padding": "2px 8px",
                        "borderRadius": "3px",
                        "fontSize": "0.62rem",
                        "fontWeight": "800",
                        "letterSpacing": "0.04em",
                        "minWidth": "22px",
                        "textAlign": "center",
                    },
                ),
                html.Div(
                    html.Div(
                        style={
                            "width": f"{bw:.0f}%",
                            "height": "5px",
                            "background": "rgba(0,0,0,0.25)",
                            "borderRadius": "3px",
                        }
                    ),
                    style={
                        "flex": "1",
                        "background": "rgba(0,0,0,0.07)",
                        "borderRadius": "3px",
                        "height": "5px",
                    },
                ),
                html.Span(
                    f"{s['prob']:.1f}%",
                    style={
                        "minWidth": "44px",
                        "textAlign": "right",
                        "fontSize": "0.85rem",
                        "fontWeight": "600",
                        "color": TDIM,
                        "fontVariantNumeric": "tabular-nums",
                    },
                ),
                html.Span(
                    american,
                    style={
                        "minWidth": "54px",
                        "textAlign": "right",
                        "fontSize": "0.82rem",
                        "fontWeight": "700",
                        "color": TEXT,
                        "fontVariantNumeric": "tabular-nums",
                        "fontFamily": BARLOW,
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "padding": "9px 0",
                "borderBottom": f"1px solid {BORDER}",
            },
        )

    return html.Div(
        [
            html.Div(
                "Top Scorelines",
                style={
                    "fontFamily": BARLOW,
                    "fontSize": "0.9rem",
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "color": TDIM,
                    "marginBottom": "12px",
                },
            ),
            *[row(s) for s in rows],
        ]
    )


def edge_rows(outcomes, btn_type=None):
    def card(label, mp, ip, idx):
        edge = mp - ip
        kelly = max(0.0, edge / (1 - ip)) * 100 if ip < 1 else 0.0
        if edge > 0.03:
            ec, bg, el = "#22C55E", "rgba(34,197,94,0.12)", "VALUE"
        elif edge > 0:
            ec, bg, el = "#F59E0B", "rgba(245,158,11,0.12)", "EDGE"
        else:
            ec, bg, el = "#EF4444", "rgba(239,68,68,0.10)", "FADE"

        right = [
            html.Div(
                f"{edge:+.1%}",
                style={
                    "fontFamily": BARLOW,
                    "fontSize": "1.5rem",
                    "fontWeight": "800",
                    "color": ec,
                    "lineHeight": "1",
                },
            ),
            html.Div(
                el,
                style={
                    "fontSize": "0.58rem",
                    "fontWeight": "700",
                    "letterSpacing": "0.08em",
                    "color": ec,
                },
            ),
        ]
        if kelly >= 0.5:
            right.append(
                html.Div(
                    f"Kelly {kelly:.0f}%",
                    style={
                        "fontSize": "0.65rem",
                        "color": TDIM,
                        "marginTop": "3px",
                    },
                )
            )

        key = f"{btn_type}-{idx}" if btn_type else None

        save_row = (
            html.Div(
                [
                    dcc.Input(
                        id={"type": "stake-input", "index": key},
                        type="number",
                        placeholder="Stake",
                        min=0,
                        step=0.01,
                        debounce=False,
                        style={
                            "flex": "1",
                            "background": "rgba(255,255,255,0.6)",
                            "border": f"1px solid {ec}44",
                            "borderRadius": "5px",
                            "color": TEXT,
                            "padding": "4px 8px",
                            "fontSize": "0.78rem",
                            "fontFamily": INTER,
                            "minWidth": "0",
                        },
                    ),
                    html.Button(
                        "+ Log",
                        id={"type": "save-bet", "index": key},
                        n_clicks=0,
                        style={
                            "background": ec,
                            "border": "none",
                            "borderRadius": "5px",
                            "color": "#fff",
                            "padding": "4px 12px",
                            "fontSize": "0.72rem",
                            "fontWeight": "700",
                            "cursor": "pointer",
                            "whiteSpace": "nowrap",
                            "letterSpacing": "0.04em",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "6px",
                    "marginTop": "8px",
                    "alignItems": "center",
                },
            )
            if key
            else html.Div()
        )

        return html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            label,
                            style={
                                "fontWeight": "600",
                                "fontSize": "0.88rem",
                                "color": TEXT,
                                "marginBottom": "3px",
                            },
                        ),
                        html.Div(
                            [
                                html.Span(
                                    [
                                        "Model ",
                                        html.B(
                                            f"{mp * 100:.1f}%", style={"color": TEXT}
                                        ),
                                    ],
                                    style={
                                        "fontSize": "0.75rem",
                                        "color": TDIM,
                                        "marginRight": "12px",
                                    },
                                ),
                                html.Span(
                                    [
                                        "Market ",
                                        html.B(
                                            f"{ip * 100:.1f}%", style={"color": TEXT}
                                        ),
                                    ],
                                    style={"fontSize": "0.75rem", "color": TDIM},
                                ),
                            ]
                        ),
                        save_row,
                    ],
                    style={"flex": "1"},
                ),
                html.Div(
                    right,
                    style={
                        "textAlign": "right",
                        "paddingRight": "14px",
                        "alignSelf": "flex-start",
                        "paddingTop": "2px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "flex-start",
                "borderLeft": f"3px solid {ec}",
                "background": bg,
                "padding": "10px 0 10px 14px",
                "marginBottom": "6px",
                "borderRadius": "0 8px 8px 0",
            },
        )

    return html.Div([card(l, m, i, idx) for idx, (l, m, i) in enumerate(outcomes)])


# ── Tab layouts ───────────────────────────────────────────────────────────────


def calc_pnl(stake, odds, result):
    if result == "pending" or stake is None:
        return None
    if result == "lost":
        return -float(stake)
    if result == "push":
        return 0.0
    stake = float(stake)
    return stake * (odds / 100) if odds > 0 else stake * (100 / abs(odds))


def bet_tracker_tab():
    return html.Div(
        [
            html.Div(id="tracker-summary"),
            html.Div(id="tracker-table", style={"marginTop": "16px"}),
        ],
        style={"padding": "20px 24px"},
    )


def predictions_tab():
    return html.Div(
        [
            html.Div(
                [
                    dcc.RadioItems(
                        id="date-mode",
                        options=[
                            {"label": "Today", "value": "today"},
                            {"label": "Tomorrow", "value": "tomorrow"},
                        ],
                        value="today",
                        inline=True,
                        inputStyle={"display": "none"},
                        labelStyle={
                            "cursor": "pointer",
                            "padding": "5px 16px",
                            "borderRadius": "20px",
                            "fontSize": "0.82rem",
                            "fontWeight": "600",
                            "color": TDIM,
                            "border": f"1px solid {BORDER}",
                            "marginRight": "6px",
                            "userSelect": "none",
                            "transition": "all 0.15s",
                        },
                        className="date-radio",
                    ),
                    dcc.Dropdown(
                        id="date-dropdown",
                        options=[],
                        placeholder="Pick a date…",
                        clearable=True,
                        style={
                            "width": "155px",
                            "fontSize": "0.82rem",
                            "fontFamily": INTER,
                        },
                        className="date-dropdown",
                    ),
                    html.Span(
                        id="fixture-count",
                        style={
                            "fontSize": "0.78rem",
                            "color": TDIM,
                            "marginLeft": "auto",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "12px",
                    "gap": "6px",
                },
            ),
            html.Div(id="fixtures-banners"),
        ],
        style={"padding": "12px 24px 20px"},
    )


def pipeline_tab():
    rows = []
    for label, script, desc in PIPELINE_STAGES:
        rows.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                label,
                                style={
                                    "fontWeight": "700",
                                    "fontSize": "0.95rem",
                                    "marginBottom": "2px",
                                    "color": TEXT,
                                },
                            ),
                            html.Div(
                                desc, style={"fontSize": "0.75rem", "color": TDIM}
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Button(
                        "▶ Run",
                        id={"type": "pipe-btn", "index": script},
                        n_clicks=0,
                        className="pipe-btn",
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "padding": "14px 16px",
                    "borderBottom": f"1px solid {BORDER}",
                },
            )
        )

    rows.append(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            "Full Pipeline",
                            style={
                                "fontWeight": "700",
                                "fontSize": "0.95rem",
                                "marginBottom": "2px",
                                "color": TEXT,
                            },
                        ),
                        html.Div(
                            "Runs all stages end-to-end",
                            style={"fontSize": "0.75rem", "color": TDIM},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                html.Button(
                    "▶ Run All",
                    id="pipe-all-btn",
                    n_clicks=0,
                    className="pipe-btn",
                    style={
                        "background": "rgba(47,129,247,0.15)",
                        "borderColor": "rgba(47,129,247,0.35)",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "padding": "14px 16px",
            },
        )
    )

    return html.Div(
        [
            html.Div(
                rows,
                style={
                    "background": BG2,
                    "borderRadius": "10px",
                    "border": f"1px solid {BORDER}",
                    "marginBottom": "14px",
                },
            ),
            dcc.Loading(html.Div(id="pipe-output"), type="dot"),
        ],
        style={"padding": "16px 20px"},
    )


def xg_editor_tab():
    return html.Div(
        [
            html.P(
                "Enter xG for WC 2026 games. Save, then run Build Features + Train Model.",
                style={"fontSize": "0.85rem", "color": TDIM, "marginBottom": "14px"},
            ),
            html.Div(id="xg-table-div"),
            html.Div(
                [
                    html.Button(
                        "💾 Save",
                        id="xg-save-btn",
                        n_clicks=0,
                        className="pipe-btn",
                        style={
                            "background": "rgba(47,129,247,0.15)",
                            "borderColor": "rgba(47,129,247,0.35)",
                        },
                    ),
                    html.Span(
                        id="xg-save-msg",
                        style={
                            "marginLeft": "12px",
                            "fontSize": "0.85rem",
                            "color": "#22C55E",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginTop": "10px"},
            ),
        ],
        style={"padding": "16px 20px"},
    )


# ── History helpers ───────────────────────────────────────────────────────────


def _date_from_gid_hist(gid):
    p = gid.split("_")
    return datetime.date(2000 + int(p[0]), int(p[1]), int(p[2]))


def _metric_card(value, label):
    return html.Div(
        [
            html.Div(
                value,
                style={
                    "fontFamily": BARLOW,
                    "fontSize": "1.6rem",
                    "fontWeight": "800",
                    "color": TEXT,
                },
            ),
            html.Div(
                label,
                style={
                    "fontSize": "0.68rem",
                    "color": TDIM,
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.06em",
                    "marginTop": "2px",
                },
            ),
        ],
        style={
            "background": BG2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "minWidth": "90px",
            "flex": "1",
        },
    )


def _team_chip(code, name, color, fg):
    return html.Div(
        [
            html.Span(
                code,
                style={
                    "background": color,
                    "color": fg,
                    "padding": "3px 9px",
                    "borderRadius": "4px",
                    "fontFamily": BARLOW,
                    "fontSize": "1rem",
                    "fontWeight": "800",
                },
            ),
            html.Span(
                name,
                style={
                    "fontSize": "0.78rem",
                    "color": TDIM,
                    "marginLeft": "8px",
                    "whiteSpace": "nowrap",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "maxWidth": "130px",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center"},
    )


def history_match_card(r):
    """Banner-style card for History tab — identical header to match_card plus a score strip."""
    left, center, right = _banner_panels(r, scale=0.68)

    ts = int(r["team_score"]) if pd.notna(r.get("team_score")) else "?"
    os_ = int(r["opp_score"]) if pd.notna(r.get("opp_score")) else "?"
    has_preds = bool(r.get("has_preds", False))

    if has_preds:
        if r.get("is_correct"):
            badge = html.Span(
                "✓  Correct",
                style={
                    "color": "#22C55E",
                    "fontWeight": "700",
                    "fontSize": "0.72rem",
                },
            )
        else:
            top = r.get("model_top_label", "")
            badge = html.Span(
                f"✗  Predicted {top}",
                style={
                    "color": "#EF4444",
                    "fontWeight": "700",
                    "fontSize": "0.72rem",
                },
            )
    else:
        badge = html.Span()

    score_strip = html.Div(
        [
            html.Span(
                "Final",
                style={
                    "fontSize": "0.68rem",
                    "color": TDIM,
                    "fontWeight": "700",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.08em",
                    "marginRight": "10px",
                },
            ),
            html.Span(
                str(ts),
                style={
                    "fontFamily": BARLOW,
                    "fontWeight": "800",
                    "fontSize": "1.05rem",
                },
            ),
            html.Span(" – ", style={"color": TDIM, "margin": "0 2px"}),
            html.Span(
                str(os_),
                style={
                    "fontFamily": BARLOW,
                    "fontWeight": "800",
                    "fontSize": "1.05rem",
                },
            ),
            badge,
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "6px",
            "background": BG2,
            "padding": "7px 14px",
            "borderTop": f"1px solid {BORDER}",
        },
    )

    return html.Div(
        [
            html.Div([left, center, right], style={"display": "flex"}),
            score_strip,
        ],
        id={"type": "match-banner", "index": r["game_id"]},
        n_clicks=0,
        className="match-card",
        style={
            "marginBottom": "10px",
            "cursor": "pointer",
            "borderRadius": "12px",
            "overflow": "hidden",
            "boxShadow": "0 3px 20px rgba(0,0,0,0.18)",
        },
    )


def history_card(r):
    t1, t2 = r["t1"], r["t2"]
    t1n, t2n = full(t1), full(t2)
    tc, oc = TEAM_COLORS.get(t1, "#444"), TEAM_COLORS.get(t2, "#444")
    tfg, ofg = _fg(tc), _fg(oc)

    t1_score = int(r["t1_score"]) if pd.notna(r.get("t1_score")) else "?"
    t2_score = int(r["t2_score"]) if pd.notna(r.get("t2_score")) else "?"

    has_preds = bool(r.get("has_preds", False))

    badge = None
    if has_preds:
        if r.get("is_correct"):
            badge = html.Span(
                "✓ Correct",
                style={
                    "color": "#22C55E",
                    "fontWeight": "700",
                    "fontSize": "0.75rem",
                },
            )
        else:
            top = r.get("model_top_label", "")
            badge = html.Span(
                f"✗ Miss  (predicted {top})",
                style={
                    "color": "#EF4444",
                    "fontWeight": "700",
                    "fontSize": "0.75rem",
                },
            )

    if has_preds:
        pw, pd_, pl = r["t1_p_win"], r["t1_p_draw"], r["t1_p_loss"]
        xg1, xg2 = r.get("t1_exp_goals", 0), r.get("t2_exp_goals", 0)
        rps = r.get("rps_score")
        rps_str = f"  ·  RPS {rps:.3f}" if pd.notna(rps) else ""
        details = html.Div(
            [
                html.Div(
                    f"{t1n} {pw * 100:.0f}%  ·  Draw {pd_ * 100:.0f}%  ·  {t2n} {pl * 100:.0f}%",
                    style={"fontSize": "0.78rem", "color": TDIM},
                ),
                html.Div(
                    f"xG  {t1n} {xg1:.2f}  ·  {t2n} {xg2:.2f}{rps_str}",
                    style={"fontSize": "0.73rem", "color": TDIM, "marginTop": "2px"},
                ),
            ],
            style={"marginTop": "8px"},
        )
    else:
        details = html.Div(
            "No predictions — run Retro Predictions in the Pipeline tab first.",
            style={"fontSize": "0.73rem", "color": TDIM, "marginTop": "8px"},
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        r.get("date_str", ""),
                        style={
                            "fontSize": "0.75rem",
                            "color": TDIM,
                            "fontWeight": "600",
                        },
                    ),
                    badge or html.Span(),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "10px",
                },
            ),
            html.Div(
                [
                    html.Div(_team_chip(t1, t1n, tc, tfg), style={"flex": "1"}),
                    html.Div(
                        [
                            html.Span(
                                str(t1_score),
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "2rem",
                                    "fontWeight": "800",
                                },
                            ),
                            html.Span(
                                " – ",
                                style={
                                    "color": TDIM,
                                    "margin": "0 4px",
                                    "fontSize": "1.2rem",
                                },
                            ),
                            html.Span(
                                str(t2_score),
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "2rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "textAlign": "center",
                            "flexShrink": "0",
                            "padding": "0 12px",
                        },
                    ),
                    html.Div(
                        _team_chip(t2, t2n, oc, ofg),
                        style={
                            "flex": "1",
                            "display": "flex",
                            "justifyContent": "flex-end",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            details,
        ],
        style={
            "background": BG2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "14px 16px",
            "marginBottom": "8px",
        },
    )


def load_history_data():
    """
    Returns (df, metrics) where df has one row per past WC2026 game with both
    match-card-compatible prediction columns and actual score / accuracy fields.
    """
    ELO_PATH = ROOT / "data" / "elo_results.csv"
    RETRO_PATH = ROOT / "data" / "retro_predictions.csv"

    if not ELO_PATH.exists():
        return pd.DataFrame(), {}

    elo = pd.read_csv(ELO_PATH)
    wc26 = elo[elo["game_id"].str.startswith("26_") & (elo["world_cup"] == 1)].copy()
    wc26 = wc26.dropna(subset=["t1_score", "t2_score"])

    if wc26.empty:
        return pd.DataFrame(), {}

    today = datetime.date.today()
    wc26["_date"] = wc26["game_id"].apply(_date_from_gid_hist)
    wc26 = wc26[wc26["_date"] < today].copy()

    if wc26.empty:
        return pd.DataFrame(), {}

    wc26["date_str"] = wc26["_date"].apply(lambda d: d.strftime("%b %d"))
    wc26["date_iso"] = wc26["_date"].apply(lambda d: d.isoformat())
    wc26["t1_score"] = wc26["t1_score"].astype(float)
    wc26["t2_score"] = wc26["t2_score"].astype(float)

    if RETRO_PATH.exists():
        retro = pd.read_csv(RETRO_PATH)
        wc26 = wc26.merge(
            retro[
                [
                    "game_id",
                    "team",
                    "exp_goals",
                    "opp_exp_goals",
                    "p_win",
                    "p_draw",
                    "p_loss",
                    "ml_score",
                ]
            ],
            left_on=["game_id", "t1"],
            right_on=["game_id", "team"],
            how="left",
        )
        wc26.drop(columns=["team"], errors="ignore", inplace=True)
        wc26["has_preds"] = wc26["p_win"].notna()
    else:
        wc26["has_preds"] = False
        for col in [
            "exp_goals",
            "opp_exp_goals",
            "p_win",
            "p_draw",
            "p_loss",
            "ml_score",
        ]:
            wc26[col] = np.nan

    # Match-card-compatible columns from t1 perspective
    wc26["team"] = wc26["t1"]
    wc26["opponent"] = wc26["t2"]
    wc26["team_score"] = wc26["t1_score"]
    wc26["opp_score"] = wc26["t2_score"]

    wc26["actual_outcome"] = np.where(
        wc26["team_score"] > wc26["opp_score"],
        "team_win",
        np.where(wc26["team_score"] == wc26["opp_score"], "draw", "opp_win"),
    )

    def _top(row):
        if not row["has_preds"]:
            return None
        opts = {
            "team_win": row["p_win"],
            "draw": row["p_draw"],
            "opp_win": row["p_loss"],
        }
        return max(opts, key=opts.get)

    wc26["model_top"] = wc26.apply(_top, axis=1)

    def _top_label(row):
        if not row["model_top"]:
            return ""
        return {
            "team_win": f"{full(row['team'])} Win",
            "draw": "Draw",
            "opp_win": f"{full(row['opponent'])} Win",
        }[row["model_top"]]

    wc26["model_top_label"] = wc26.apply(_top_label, axis=1)
    wc26["is_correct"] = wc26["model_top"] == wc26["actual_outcome"]
    wc26.loc[~wc26["has_preds"], "is_correct"] = None

    def _rps(row):
        if not row["has_preds"]:
            return np.nan
        code = {"team_win": 1, "draw": 0, "opp_win": -1}[row["actual_outcome"]]
        probs = [row["p_win"], row["p_draw"], row["p_loss"]]
        cum = np.cumsum(probs)
        actv = np.cumsum([code == 1, code == 0, code == -1])
        return float(np.mean((cum - actv) ** 2))

    wc26["rps_score"] = wc26.apply(_rps, axis=1)

    with_preds = wc26[wc26["has_preds"]]
    metrics = {
        "total_games": len(wc26),
        "with_preds": len(with_preds),
        "correct": int(with_preds["is_correct"].sum()) if len(with_preds) else 0,
        "accuracy": float(with_preds["is_correct"].mean()) if len(with_preds) else 0.0,
        "avg_rps": float(with_preds["rps_score"].mean()) if len(with_preds) else 0.0,
    }

    wc26 = wc26.sort_values("_date", ascending=False).reset_index(drop=True)
    return wc26, metrics


def history_tab():
    return html.Div(
        [
            html.Div(
                [
                    dcc.RadioItems(
                        id="history-date-mode",
                        options=[{"label": "All", "value": "all"}],
                        value="all",
                        inline=True,
                        inputStyle={"display": "none"},
                        labelStyle={
                            "cursor": "pointer",
                            "padding": "5px 16px",
                            "borderRadius": "20px",
                            "fontSize": "0.82rem",
                            "fontWeight": "600",
                            "color": TDIM,
                            "border": f"1px solid {BORDER}",
                            "marginRight": "6px",
                            "userSelect": "none",
                            "transition": "all 0.15s",
                        },
                        className="date-radio",
                    ),
                    dcc.Dropdown(
                        id="history-date-dropdown",
                        options=[],
                        placeholder="Pick a date…",
                        clearable=True,
                        style={
                            "width": "155px",
                            "fontSize": "0.82rem",
                            "fontFamily": INTER,
                        },
                        className="date-dropdown",
                    ),
                    html.Span(
                        id="history-game-count",
                        style={
                            "fontSize": "0.78rem",
                            "color": TDIM,
                            "marginLeft": "auto",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "marginBottom": "12px",
                    "gap": "6px",
                },
            ),
            dcc.Loading(html.Div(id="history-content"), type="dot"),
        ],
        style={"padding": "12px 24px 20px"},
    )


# ── App ───────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="WC 2026 Predictions",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

_label_style = {
    "fontSize": "0.7rem",
    "color": TDIM,
    "fontWeight": "700",
    "marginBottom": "4px",
    "display": "block",
    "textTransform": "uppercase",
    "letterSpacing": "0.06em",
}

_input_style = {
    "width": "100%",
    "background": BG,
    "border": f"1px solid {BORDER}",
    "borderRadius": "8px",
    "color": TEXT,
    "padding": "9px 12px",
    "fontSize": "0.9rem",
    "fontFamily": INTER,
    "boxSizing": "border-box",
}

login_modal = html.Div(
    [
        html.Div(
            id="login-backdrop",
            n_clicks=0,
            style={
                "position": "fixed",
                "inset": "0",
                "background": "rgba(0,0,0,0.75)",
                "backdropFilter": "blur(3px)",
                "zIndex": "300",
                "cursor": "pointer",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Button(
                            "✕",
                            id="login-close",
                            n_clicks=0,
                            className="modal-close-btn",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "flex-end",
                        "padding": "12px 16px 0",
                    },
                ),
                html.Div(
                    [
                        html.Div(
                            "Admin Login",
                            style={
                                "fontFamily": BARLOW,
                                "fontSize": "1.5rem",
                                "fontWeight": "800",
                                "letterSpacing": "0.03em",
                                "color": TEXT,
                                "marginBottom": "20px",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Username", style=_label_style),
                                dcc.Input(
                                    id="login-username",
                                    value="admin",
                                    disabled=True,
                                    style={
                                        **_input_style,
                                        "opacity": "0.55",
                                        "cursor": "not-allowed",
                                    },
                                ),
                            ],
                            style={"marginBottom": "14px"},
                        ),
                        html.Div(
                            [
                                html.Label("Password", style=_label_style),
                                dcc.Input(
                                    id="login-password",
                                    type="password",
                                    placeholder="Enter password",
                                    debounce=False,
                                    n_submit=0,
                                    style=_input_style,
                                ),
                            ],
                            style={"marginBottom": "8px"},
                        ),
                        html.Div(
                            id="login-error",
                            style={
                                "color": "#EF4444",
                                "fontSize": "0.82rem",
                                "minHeight": "20px",
                                "marginBottom": "12px",
                            },
                        ),
                        html.Button(
                            "Login",
                            id="login-submit",
                            n_clicks=0,
                            style={
                                "width": "100%",
                                "background": ACCENT,
                                "border": "none",
                                "borderRadius": "8px",
                                "color": "#fff",
                                "padding": "10px",
                                "fontSize": "0.92rem",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "letterSpacing": "0.03em",
                            },
                        ),
                    ],
                    style={"padding": "4px 24px 28px"},
                ),
            ],
            style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "background": BG2,
                "borderRadius": "16px",
                "width": "min(360px, 90vw)",
                "zIndex": "301",
                "boxShadow": "0 24px 80px rgba(0,0,0,0.7)",
                "border": f"1px solid {BORDER}",
            },
        ),
    ],
    id="login-modal",
    style={"display": "none"},
)

modal = html.Div(
    [
        # Backdrop
        html.Div(
            id="modal-backdrop",
            n_clicks=0,
            style={
                "position": "fixed",
                "inset": "0",
                "background": "rgba(0,0,0,0.75)",
                "backdropFilter": "blur(3px)",
                "zIndex": "200",
                "cursor": "pointer",
            },
        ),
        # Panel
        html.Div(
            [
                # Close row
                html.Div(
                    [
                        html.Button(
                            "✕",
                            id="modal-close",
                            n_clicks=0,
                            className="modal-close-btn",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "flex-end",
                        "padding": "12px 16px 0",
                    },
                ),
                # Match header
                html.Div(id="modal-match-header-div", style={"padding": "0 20px 4px"}),
                # Tabs
                dcc.Tabs(
                    id="modal-tabs",
                    value="betting",
                    className="modal-tabs",
                    children=[
                        dcc.Tab(
                            label="Betting Calculator",
                            value="betting",
                            className="modal-tab",
                            selected_className="modal-tab--selected",
                            children=[
                                html.Div(
                                    [
                                        # Market radio
                                        html.Div(
                                            [
                                                html.Span(
                                                    "Market",
                                                    style={
                                                        "fontFamily": BARLOW,
                                                        "fontSize": "0.85rem",
                                                        "fontWeight": "700",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "0.08em",
                                                        "color": TDIM,
                                                        "marginRight": "12px",
                                                    },
                                                ),
                                                dcc.RadioItems(
                                                    id="market-mode",
                                                    options=[
                                                        {
                                                            "label": "W/D/L ML",
                                                            "value": "1x2",
                                                        },
                                                        {
                                                            "label": "DRAW NO BET",
                                                            "value": "dnb",
                                                        },
                                                    ],
                                                    value="1x2",
                                                    inline=True,
                                                    inputStyle={"display": "none"},
                                                    labelStyle={
                                                        "cursor": "pointer",
                                                        "padding": "4px 12px",
                                                        "borderRadius": "6px",
                                                        "fontSize": "0.78rem",
                                                        "fontWeight": "700",
                                                        "color": TDIM,
                                                        "border": f"1px solid {BORDER}",
                                                        "marginLeft": "6px",
                                                        "userSelect": "none",
                                                        "transition": "all 0.15s",
                                                    },
                                                    className="date-radio",
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "alignItems": "center",
                                                "marginBottom": "16px",
                                            },
                                        ),
                                        # Odds inputs
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            id="odds-win-label",
                                                            style=_label_style,
                                                        ),
                                                        dcc.Input(
                                                            id="odds-win",
                                                            type="number",
                                                            value=-110,
                                                            step=1,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="odds-win-impl",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Draw", style=_label_style
                                                        ),
                                                        dcc.Input(
                                                            id="odds-draw",
                                                            type="number",
                                                            value=240,
                                                            step=1,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="odds-draw-impl",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    id="draw-input-wrapper",
                                                    style={"flex": "1"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            id="odds-loss-label",
                                                            style=_label_style,
                                                        ),
                                                        dcc.Input(
                                                            id="odds-loss",
                                                            type="number",
                                                            value=300,
                                                            step=1,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="odds-loss-impl",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "gap": "8px",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            id="vig-caption",
                                            style={
                                                "fontSize": "0.73rem",
                                                "color": TDIM,
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        html.Div(id="edge-rows-div"),
                                    ],
                                    style={"padding": "16px 20px 20px"},
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Total Goals",
                            value="totals",
                            className="modal-tab",
                            selected_className="modal-tab--selected",
                            children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Line", style=_label_style
                                                        ),
                                                        dcc.Input(
                                                            id="ou-line",
                                                            type="number",
                                                            value=2.5,
                                                            step=0.5,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="ou-line-model",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"flex": "0 0 80px"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Over Odds",
                                                            style=_label_style,
                                                        ),
                                                        dcc.Input(
                                                            id="odds-over",
                                                            type="number",
                                                            value=-110,
                                                            step=1,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="odds-over-impl",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Under Odds",
                                                            style=_label_style,
                                                        ),
                                                        dcc.Input(
                                                            id="odds-under",
                                                            type="number",
                                                            value=-110,
                                                            step=1,
                                                            debounce=False,
                                                            className="odds-input",
                                                            style={"width": "100%"},
                                                        ),
                                                        html.Div(
                                                            id="odds-under-impl",
                                                            style={
                                                                "fontSize": "0.67rem",
                                                                "color": TDIM,
                                                                "textAlign": "center",
                                                                "marginTop": "3px",
                                                            },
                                                        ),
                                                    ],
                                                    style={"flex": "1"},
                                                ),
                                            ],
                                            style={
                                                "display": "flex",
                                                "gap": "8px",
                                                "marginBottom": "8px",
                                            },
                                        ),
                                        html.Div(
                                            id="ou-vig-caption",
                                            style={
                                                "fontSize": "0.73rem",
                                                "color": TDIM,
                                                "marginBottom": "12px",
                                            },
                                        ),
                                        html.Div(id="ou-results-div"),
                                    ],
                                    style={"padding": "16px 20px 20px"},
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Scorelines",
                            value="scorelines",
                            className="modal-tab",
                            selected_className="modal-tab--selected",
                            children=[
                                html.Div(
                                    id="modal-scorelines-div",
                                    style={"padding": "16px 20px 20px"},
                                )
                            ],
                        ),
                        dcc.Tab(
                            label="Parlay",
                            value="parlay",
                            className="modal-tab",
                            selected_className="modal-tab--selected",
                            children=[
                                html.Div(
                                    [
                                        html.Div(id="parlay-legs-div"),
                                        html.Div(
                                            id="parlay-summary-div",
                                            style={"marginTop": "10px"},
                                        ),
                                        html.Div(
                                            [
                                                html.Label(
                                                    "Market parlay odds",
                                                    style=_label_style,
                                                ),
                                                dcc.Input(
                                                    id="parlay-market-odds",
                                                    type="number",
                                                    placeholder="+550",
                                                    step=1,
                                                    debounce=True,
                                                    className="odds-input",
                                                    style={"width": "110px"},
                                                ),
                                            ],
                                            id="parlay-odds-input-row",
                                            style={
                                                "display": "none",
                                                "marginTop": "12px",
                                            },
                                        ),
                                        html.Div(
                                            id="parlay-edge-display",
                                            style={"marginTop": "8px"},
                                        ),
                                    ],
                                    style={"padding": "16px 20px 20px"},
                                )
                            ],
                        ),
                    ],
                ),
            ],
            style={
                "position": "fixed",
                "top": "50%",
                "left": "50%",
                "transform": "translate(-50%, -50%)",
                "background": BG2,
                "borderRadius": "16px",
                "width": "min(600px, 94vw)",
                "maxHeight": "88vh",
                "overflowY": "auto",
                "zIndex": "201",
                "boxShadow": "0 24px 80px rgba(0,0,0,0.7)",
                "border": f"1px solid {BORDER}",
            },
        ),
    ],
    id="match-modal",
    style={"display": "none"},
)


_btn_header = {
    "background": "rgba(255,255,255,0.1)",
    "border": "1px solid rgba(255,255,255,0.2)",
    "borderRadius": "6px",
    "color": "rgba(255,255,255,0.8)",
    "cursor": "pointer",
    "padding": "5px 14px",
    "fontSize": "0.8rem",
    "fontWeight": "600",
    "fontFamily": INTER,
    "letterSpacing": "0.03em",
}

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="auth-store", storage_type="session", data=False),
        dcc.Store(id="pred-store"),
        dcc.Store(id="game-df-store"),
        dcc.Store(id="selected-game"),
        dcc.Store(id="bets-store", storage_type="local", data=[]),
        dcc.Store(id="parlay-store", data={}),
        dcc.Store(id="history-game-df-store"),
        html.Div(
            id="bet-toast",
            style={
                "position": "fixed",
                "bottom": "28px",
                "left": "50%",
                "transform": "translateX(-50%)",
                "zIndex": "10000",
                "pointerEvents": "none",
            },
        ),
        # Header
        html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            "⚽", style={"marginRight": "10px", "fontSize": "1.3rem"}
                        ),
                        html.Span(
                            "World Cup",
                            style={
                                "fontFamily": BARLOW,
                                "fontSize": "1.7rem",
                                "fontWeight": "800",
                                "letterSpacing": "0.05em",
                                "color": "#FFFFFF",
                            },
                        ),
                        html.Span(
                            "2026",
                            style={
                                "fontFamily": BARLOW,
                                "fontSize": "1.7rem",
                                "fontWeight": "800",
                                "letterSpacing": "0.05em",
                                "color": "rgba(255,255,255,0.45)",
                                "marginLeft": "8px",
                            },
                        ),
                        html.Span(
                            "Match Predictions",
                            style={
                                "fontFamily": INTER,
                                "fontSize": "0.82rem",
                                "color": "rgba(255,255,255,0.45)",
                                "marginLeft": "14px",
                                "fontWeight": "500",
                            },
                        ),
                    ],
                    style={"display": "flex", "alignItems": "center"},
                ),
                html.Div(
                    [
                        html.Button(
                            "🔐 Admin",
                            id="login-btn",
                            n_clicks=0,
                            title="Admin login",
                            style=_btn_header,
                        ),
                        html.Button(
                            "Logout",
                            id="logout-btn",
                            n_clicks=0,
                            title="Log out of admin",
                            style={**_btn_header, "display": "none"},
                        ),
                    ],
                    style={"display": "flex", "gap": "6px", "alignItems": "center"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "space-between",
                "padding": "14px 24px",
                "background": "#0F172A",
            },
        ),
        # Tabs — children rendered dynamically based on auth state
        dcc.Tabs(
            id="main-tabs",
            value="predictions",
            className="dash-tabs",
            children=[],
        ),
        login_modal,
        modal,
    ],
    style={"background": BG, "minHeight": "100vh", "color": TEXT, "fontFamily": INTER},
)


# ── Callbacks ─────────────────────────────────────────────────────────────────


@app.callback(
    Output("pred-store", "data"),
    Output("game-df-store", "data"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def load_data(_):
    try:
        preds = load_preds()
        gdf = build_game_df(preds)
        # Merge past WC2026 predictions so modal callbacks work for history games
        try:
            hist_df, _ = load_history_data()
            if hist_df is not None and not hist_df.empty:
                has_p = hist_df[hist_df["has_preds"].astype(bool)]
                if not has_p.empty:
                    hist_base = has_p[
                        [
                            "game_id",
                            "team",
                            "opponent",
                            "date_str",
                            "date_iso",
                            "exp_goals",
                            "opp_exp_goals",
                            "p_win",
                            "p_draw",
                            "p_loss",
                        ]
                    ].copy()
                    gdf = (
                        pd.concat([gdf, hist_base])
                        .drop_duplicates(subset="game_id", keep="first")
                        .reset_index(drop=True)
                    )
        except Exception:
            pass
        return preds.to_json(orient="split"), gdf.to_json(orient="split")
    except Exception:
        return None, None


# ── Auth callbacks ────────────────────────────────────────────────────────────


@app.callback(
    Output("main-tabs", "children"),
    Output("main-tabs", "value"),
    Input("auth-store", "data"),
    State("main-tabs", "value"),
)
def render_tabs(is_admin, current_tab):
    def _tab(label, value, content):
        return dcc.Tab(
            label=label,
            value=value,
            className="dash-tab",
            selected_className="dash-tab--selected",
            children=content,
        )

    tabs = [
        _tab("Predictions", "predictions", predictions_tab()),
        _tab("Bet Tracker", "bet-tracker", bet_tracker_tab()),
        _tab("History", "history", history_tab()),
    ]
    if is_admin:
        tabs += [
            _tab("Pipeline", "pipeline", pipeline_tab()),
            _tab("xG Editor", "xg-editor", xg_editor_tab()),
        ]

    tab_val = current_tab or "predictions"
    if not is_admin and tab_val in ("pipeline", "xg-editor"):
        tab_val = "predictions"
    return tabs, tab_val


@app.callback(
    Output("login-btn", "style"),
    Output("logout-btn", "style"),
    Input("auth-store", "data"),
)
def update_header_buttons(is_admin):
    show = _btn_header
    hide = {**_btn_header, "display": "none"}
    if is_admin:
        return hide, show
    return show, hide


@app.callback(
    Output("login-modal", "style"),
    Input("login-btn", "n_clicks"),
    Input("login-close", "n_clicks"),
    Input("login-backdrop", "n_clicks"),
    Input("auth-store", "data"),
    prevent_initial_call=True,
)
def toggle_login_modal(open_clicks, close_clicks, backdrop_clicks, is_admin):
    trigger = callback_context.triggered[0]["prop_id"]
    if "login-btn" in trigger:
        return {"display": "block"}
    if "auth-store" in trigger and is_admin:
        return {"display": "none"}  # close on successful login
    return {"display": "none"}


@app.callback(
    Output("auth-store", "data"),
    Output("login-error", "children"),
    Input("login-submit", "n_clicks"),
    Input("login-password", "n_submit"),
    Input("logout-btn", "n_clicks"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def handle_auth(submit_clicks, pw_submit, logout_clicks, password):
    trigger = callback_context.triggered[0]["prop_id"]
    if not callback_context.triggered[0]["value"]:
        raise PreventUpdate
    if "logout-btn" in trigger:
        return False, ""
    # login attempt
    if password and hmac.compare_digest(password, ADMIN_PASSWORD):
        return True, ""
    return no_update, "Incorrect password."


@app.callback(
    Output("login-password", "value"),
    Input("auth-store", "data"),
    prevent_initial_call=True,
)
def clear_password_on_login(is_admin):
    return "" if is_admin else no_update


# ── Predictions / fixture callbacks ───────────────────────────────────────────


@app.callback(
    Output("fixtures-banners", "children"),
    Output("fixture-count", "children"),
    Input("date-mode", "value"),
    Input("date-dropdown", "value"),
    Input("game-df-store", "data"),
)
def update_fixtures(mode, dropdown_date, store_json):
    if not store_json:
        return [
            html.Div(
                "No data — run Pipeline → Train Model first.",
                style={"color": TDIM, "padding": "16px"},
            )
        ], ""

    gdf = pd.read_json(StringIO(store_json), orient="split")
    today = datetime.date.today()
    tom = today + datetime.timedelta(days=1)
    gdf["_date"] = pd.to_datetime(gdf["date_iso"]).dt.date

    if dropdown_date:
        try:
            pick = datetime.date.fromisoformat(dropdown_date)
            filt = gdf[gdf["_date"] == pick]
            label = pick.strftime("%b %d")
        except Exception:
            filt = pd.DataFrame()
            label = dropdown_date
    elif mode == "today":
        filt = gdf[gdf["_date"] == today]
        label = today.strftime("%b %d")
    else:
        filt = gdf[gdf["_date"] == tom]
        label = tom.strftime("%b %d")

    filt = filt.reset_index(drop=True)

    if filt.empty:
        return [
            html.Div(
                f"No fixtures for {label}.", style={"color": TDIM, "padding": "8px"}
            )
        ], "0 fixtures"

    cards = [match_card(r) for _, r in filt.iterrows()]
    count = f"{len(filt)} fixture{'s' if len(filt) != 1 else ''}"
    return cards, count


@app.callback(
    Output("date-dropdown", "options"),
    Input("game-df-store", "data"),
)
def populate_date_dropdown(store_json):
    if not store_json:
        return []
    gdf = pd.read_json(StringIO(store_json), orient="split")
    gdf["_date"] = pd.to_datetime(gdf["date_iso"]).dt.date
    dates = sorted(gdf["_date"].unique())
    return [{"label": d.strftime("%a, %b %d"), "value": d.isoformat()} for d in dates]


@app.callback(
    Output("date-dropdown", "value"),
    Input("date-mode", "value"),
    prevent_initial_call=True,
)
def clear_dropdown_on_radio(_):
    return None


@app.callback(
    Output("selected-game", "data"),
    Input({"type": "match-banner", "index": ALL}, "n_clicks"),
    Input("modal-close", "n_clicks"),
    Input("modal-backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def handle_modal_toggle(banner_clicks, close_clicks, backdrop_clicks):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"]

    if "modal-close" in trigger or "modal-backdrop" in trigger:
        return None

    if ctx.triggered[0].get("value", 0) == 0:
        raise PreventUpdate

    try:
        info = json.loads(trigger.split(".")[0])
        return info["index"]
    except Exception:
        raise PreventUpdate


@app.callback(
    Output("match-modal", "style"),
    Input("selected-game", "data"),
)
def toggle_modal(game_id):
    if game_id:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("modal-match-header-div", "children"),
    Input("selected-game", "data"),
    State("game-df-store", "data"),
)
def update_modal_header(game_id, store_json):
    if not game_id or not store_json:
        return html.Div()
    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
    return modal_match_header(row)


@app.callback(
    Output("modal-scorelines-div", "children"),
    Input("selected-game", "data"),
    Input("modal-tabs", "value"),
    State("game-df-store", "data"),
)
def update_scorelines(game_id, tab, store_json):
    if tab != "scorelines" or not game_id or not store_json:
        return html.Div()
    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
    tc = TEAM_COLORS.get(row["team"], "#444")
    oc = TEAM_COLORS.get(row["opponent"], "#444")
    return scorelines_panel(row["exp_goals"], row["opp_exp_goals"], tc, oc)


@app.callback(
    Output("odds-win-label", "children"),
    Output("odds-loss-label", "children"),
    Input("selected-game", "data"),
    State("game-df-store", "data"),
)
def update_odds_labels(game_id, store_json):
    if not game_id or not store_json:
        return "Win", "Loss"
    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
    return f"{row['team']} Win", f"{row['opponent']} Win"


@app.callback(
    Output("draw-input-wrapper", "style"),
    Input("market-mode", "value"),
)
def toggle_draw_input(market):
    if market == "dnb":
        return {"flex": "1", "display": "none"}
    return {"flex": "1"}


@app.callback(
    Output("odds-win-impl", "children"),
    Output("odds-draw-impl", "children"),
    Output("odds-loss-impl", "children"),
    Input("odds-win", "value"),
    Input("odds-draw", "value"),
    Input("odds-loss", "value"),
)
def update_impl_displays(wo, do, lo):
    def fmt(v, default):
        try:
            return f"impl {american_to_implied(int(v or default)) * 100:.1f}%"
        except Exception:
            return ""

    return fmt(wo, -110), fmt(do, 240), fmt(lo, 300)


@app.callback(
    Output("edge-rows-div", "children"),
    Output("vig-caption", "children"),
    Input("odds-win", "value"),
    Input("odds-draw", "value"),
    Input("odds-loss", "value"),
    Input("market-mode", "value"),
    State("selected-game", "data"),
    State("game-df-store", "data"),
)
def update_edges(wo, do, lo, market, game_id, store_json):
    if not game_id or not store_json:
        return html.Div(), ""

    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
    pw, pd_, pl = row["p_win"], row["p_draw"], row["p_loss"]
    t, o = full(row["team"]), full(row["opponent"])

    wo = int(wo or -110)
    lo = int(lo or 300)

    if market == "1x2":
        do = int(do or 240)
        iw = american_to_implied(wo)
        id_ = american_to_implied(do)
        il = american_to_implied(lo)
        vig = (iw + id_ + il - 1) * 100
        outcomes = [
            (f"{t} Win", pw, iw, wo),
            ("Draw", pd_, id_, do),
            (f"{o} Win", pl, il, lo),
        ]
        caption = f"Vig {vig:.1f}%"
    else:
        iw = american_to_implied(wo)
        il = american_to_implied(lo)
        mw = pw / (pw + pl)
        ml = pl / (pw + pl)
        outcomes = [(f"{t} Win", mw, iw, wo), (f"{o} Win", ml, il, lo)]
        caption = f"Draw {pd_ * 100:.1f}% — stake refunded on draw"

    value_n = sum(1 for _, mp, ip, _ in outcomes if mp - ip > 0.03)
    edge_n = sum(1 for _, mp, ip, _ in outcomes if 0 < mp - ip <= 0.03)
    parts = []
    if value_n:
        parts.append(f"{value_n} value")
    if edge_n:
        parts.append(f"{edge_n} edge")
    if parts:
        caption += "  ·  " + "  ·  ".join(parts)

    return edge_rows([(l, m, i) for l, m, i, _ in outcomes], btn_type="ml"), caption


@app.callback(
    Output("ou-results-div", "children"),
    Output("ou-vig-caption", "children"),
    Output("ou-line-model", "children"),
    Output("odds-over-impl", "children"),
    Output("odds-under-impl", "children"),
    Input("ou-line", "value"),
    Input("odds-over", "value"),
    Input("odds-under", "value"),
    State("selected-game", "data"),
    State("game-df-store", "data"),
)
def update_ou(line, over_o, under_o, game_id, store_json):
    empty = html.Div(), "", "", "", ""
    if not game_id or not store_json:
        return empty

    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
    lam = row["exp_goals"] + row["opp_exp_goals"]
    t, o = full(row["team"]), full(row["opponent"])

    line = float(line or 2.5)
    over_o = int(over_o or -110)
    under_o = int(under_o or -110)

    p_over, p_under, p_push = ou_probs(lam, line)

    model_str = f"model {p_over * 100:.1f}% over"
    if p_push > 0.001:
        model_str += f"  ·  {p_push * 100:.1f}% push"

    io_ = american_to_implied(over_o)
    iu = american_to_implied(under_o)
    vig = (io_ + iu - 1) * 100

    over_impl = f"impl {io_ * 100:.1f}%"
    under_impl = f"impl {iu * 100:.1f}%"

    outcomes = [
        (f"Over {line}", p_over, io_, over_o),
        (f"Under {line}", p_under, iu, under_o),
    ]

    value_n = sum(1 for _, mp, ip, _ in outcomes if mp - ip > 0.03)
    edge_n = sum(1 for _, mp, ip, _ in outcomes if 0 < mp - ip <= 0.03)
    caption = f"Vig {vig:.1f}%"
    parts = []
    if value_n:
        parts.append(f"{value_n} value")
    if edge_n:
        parts.append(f"{edge_n} edge")
    if parts:
        caption += "  ·  " + "  ·  ".join(parts)

    return (
        edge_rows([(l, m, i) for l, m, i, _ in outcomes], btn_type="ou"),
        caption,
        model_str,
        over_impl,
        under_impl,
    )


@app.callback(
    Output("bets-store", "data"),
    Output("bet-toast", "children"),
    Input({"type": "save-bet", "index": ALL}, "n_clicks"),
    Input({"type": "result-btn", "index": ALL}, "n_clicks"),
    Input({"type": "delete-bet-btn", "index": ALL}, "n_clicks"),
    State({"type": "stake-input", "index": ALL}, "value"),
    State("odds-win", "value"),
    State("odds-draw", "value"),
    State("odds-loss", "value"),
    State("odds-over", "value"),
    State("odds-under", "value"),
    State("ou-line", "value"),
    State("market-mode", "value"),
    State("selected-game", "data"),
    State("game-df-store", "data"),
    State("bets-store", "data"),
    prevent_initial_call=True,
)
def manage_bets(
    save_clicks,
    result_clicks,
    delete_clicks,
    stake_vals,
    wo,
    do,
    lo,
    over_o,
    under_o,
    ou_line,
    market,
    game_id,
    store_json,
    current_bets,
):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"]
    bets = list(current_bets or [])

    # ── Result / Delete (tracker table buttons) ───────────────────────────────
    if "result-btn" in trigger and ctx.triggered[0]["value"]:
        info = json.loads(trigger.split(".")[0])
        parts = info["index"].split("|")
        bet_id, result = parts[0], parts[1]
        for b in bets:
            if b["id"] == bet_id:
                b["result"] = result
                break
        return bets, no_update

    if "delete-bet-btn" in trigger and ctx.triggered[0]["value"]:
        info = json.loads(trigger.split(".")[0])
        bet_id = info["index"]
        bets = [b for b in bets if b["id"] != bet_id]
        return bets, no_update

    # ── Inline save buttons on edge cards ─────────────────────────────────────
    if "save-bet" in trigger and ctx.triggered[0]["value"]:
        if not game_id or not store_json:
            raise PreventUpdate

        # Identify which button fired
        btn_info = json.loads(trigger.split(".")[0])
        btn_index = btn_info["index"]  # e.g. "ml-0" or "ou-2"
        btn_type, pos_str = btn_index.rsplit("-", 1)
        pos = int(pos_str)

        stake = None
        for comp in ctx.states_list[0]:
            if comp["id"]["index"] == btn_index:
                stake = comp.get("value")
                break
        if stake is None:
            raise PreventUpdate

        gdf = pd.read_json(StringIO(store_json), orient="split")
        row = gdf[gdf["game_id"] == game_id].iloc[0].to_dict()
        t, o = full(row["team"]), full(row["opponent"])
        pw, pd_, pl = row["p_win"], row["p_draw"], row["p_loss"]

        wo_ = int(wo or -110)
        lo_ = int(lo or 300)
        over_o_ = int(over_o or -110)
        under_o_ = int(under_o or -110)
        line_ = float(ou_line or 2.5)

        if btn_type == "ml":
            if market == "1x2":
                do_ = int(do or 240)
                iw = american_to_implied(wo_)
                id_ = american_to_implied(do_)
                il = american_to_implied(lo_)
                outcomes = [
                    (f"{t} Win", pw, iw, wo_),
                    ("Draw", pd_, id_, do_),
                    (f"{o} Win", pl, il, lo_),
                ]
            else:
                iw = american_to_implied(wo_)
                il = american_to_implied(lo_)
                mw = pw / (pw + pl)
                ml_ = pl / (pw + pl)
                outcomes = [
                    (f"{t} Win", mw, iw, wo_),
                    (f"{o} Win", ml_, il, lo_),
                ]
        else:  # ou
            p_over, p_under, _ = ou_probs(
                row["exp_goals"] + row["opp_exp_goals"], line_
            )
            io_ = american_to_implied(over_o_)
            iu = american_to_implied(under_o_)
            outcomes = [
                (f"Over {line_}", p_over, io_, over_o_),
                (f"Under {line_}", p_under, iu, under_o_),
            ]

        if pos >= len(outcomes):
            raise PreventUpdate

        label, mp, ip, odds = outcomes[pos]
        edge = mp - ip
        bet = {
            "id": str(int(datetime.datetime.now().timestamp() * 1000)),
            "game_id": game_id,
            "match": f"{t} vs {o}",
            "date": row["date_str"],
            "selection": label,
            "odds": odds,
            "stake": float(stake),
            "model_prob": round(mp, 4),
            "market_prob": round(ip, 4),
            "edge": round(edge, 4),
            "result": "pending",
        }
        bets.append(bet)
        ts = bet["id"]
        toast = html.Div(
            [
                html.Span("✓ ", style={"fontWeight": "700", "color": "#22C55E"}),
                f"Bet logged — {label} @ {odds:+d}",
            ],
            id=f"toast-inner-{ts}",
            className="bet-toast-inner",
        )
        return bets, toast

    raise PreventUpdate


@app.callback(
    Output("tracker-summary", "children"),
    Output("tracker-table", "children"),
    Input("bets-store", "data"),
)
def render_tracker(bets):
    bets = bets or []

    if not bets:
        return html.Div(), html.Div(
            "No bets logged yet. Open a game and tap + Log on any edge card.",
            style={"color": TDIM, "fontSize": "0.88rem", "padding": "24px 0"},
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    settled = [b for b in bets if b["result"] != "pending"]
    pnl_vals = [calc_pnl(b["stake"], b["odds"], b["result"]) for b in settled]
    pnl_vals = [v for v in pnl_vals if v is not None]
    total_staked_settled = sum(b["stake"] for b in settled)
    total_pnl = sum(pnl_vals)
    roi = (total_pnl / total_staked_settled * 100) if total_staked_settled else 0
    wins = sum(1 for b in settled if b["result"] == "won")
    losses = sum(1 for b in settled if b["result"] == "lost")
    pushes = sum(1 for b in settled if b["result"] == "push")
    pnl_color = "#22C55E" if total_pnl >= 0 else "#EF4444"

    def stat(label, val, color=TEXT):
        return html.Div(
            [
                html.Div(
                    val,
                    style={
                        "fontFamily": BARLOW,
                        "fontSize": "1.5rem",
                        "fontWeight": "800",
                        "color": color,
                        "lineHeight": "1",
                    },
                ),
                html.Div(
                    label,
                    style={
                        "fontSize": "0.68rem",
                        "color": TDIM,
                        "marginTop": "3px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.08em",
                        "fontWeight": "600",
                    },
                ),
            ],
            style={"textAlign": "center", "flex": "1"},
        )

    pnl_str = f"{total_pnl:+.2f}"
    roi_str = f"{roi:+.1f}%"
    summary = html.Div(
        [
            stat(f"Bets ({len(bets)})", str(len(settled)) + " settled"),
            html.Div(style={"width": "1px", "background": BORDER, "margin": "0 8px"}),
            stat("P&L", pnl_str, pnl_color),
            html.Div(style={"width": "1px", "background": BORDER, "margin": "0 8px"}),
            stat("ROI", roi_str, pnl_color),
            html.Div(style={"width": "1px", "background": BORDER, "margin": "0 8px"}),
            stat("Record", f"{wins}–{losses}–{pushes}"),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "background": BG2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "padding": "16px 20px",
        },
    )

    # ── Table ─────────────────────────────────────────────────────────────────
    RESULTS = ["pending", "won", "lost", "push"]
    RC = {"pending": TDIM, "won": "#22C55E", "lost": "#EF4444", "push": "#F59E0B"}
    RBG = {
        "pending": "transparent",
        "won": "rgba(34,197,94,0.12)",
        "lost": "rgba(239,68,68,0.10)",
        "push": "rgba(245,158,11,0.10)",
    }

    header = html.Div(
        [
            html.Div("Date", style={"flex": "0 0 54px"}),
            html.Div("Match", style={"flex": "2"}),
            html.Div("Bet", style={"flex": "2"}),
            html.Div("Odds", style={"flex": "0 0 54px", "textAlign": "right"}),
            html.Div("Stake", style={"flex": "0 0 60px", "textAlign": "right"}),
            html.Div("Edge", style={"flex": "0 0 58px", "textAlign": "right"}),
            html.Div("Result", style={"flex": "3", "textAlign": "center"}),
            html.Div("P&L", style={"flex": "0 0 60px", "textAlign": "right"}),
            html.Div("", style={"flex": "0 0 28px"}),
        ],
        style={
            "display": "flex",
            "alignItems": "center",
            "gap": "6px",
            "padding": "6px 12px",
            "fontSize": "0.65rem",
            "fontWeight": "700",
            "color": TDIM,
            "textTransform": "uppercase",
            "letterSpacing": "0.07em",
            "borderBottom": f"1px solid {BORDER}",
        },
    )

    def bet_row(b):
        pnl = calc_pnl(b["stake"], b["odds"], b["result"])
        pnl_disp = f"{pnl:+.2f}" if pnl is not None else "—"
        pnl_col = (
            "#22C55E" if (pnl or 0) > 0 else ("#EF4444" if (pnl or 0) < 0 else TDIM)
        )
        edge_col = (
            "#22C55E"
            if b["edge"] > 0.03
            else ("#F59E0B" if b["edge"] > 0 else "#EF4444")
        )
        odds_str = (
            prob_to_american(b["market_prob"] * 100)
            if b["odds"] == 0
            else (f"+{b['odds']}" if b["odds"] > 0 else str(b["odds"]))
        )

        result_btns = html.Div(
            [
                html.Button(
                    r.upper(),
                    n_clicks=0,
                    id={"type": "result-btn", "index": f"{b['id']}|{r}"},
                    style={
                        "padding": "2px 7px",
                        "fontSize": "0.6rem",
                        "fontWeight": "700",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "border": f"1px solid {RC[r]}",
                        "background": RBG[r] if b["result"] == r else "transparent",
                        "color": RC[r] if b["result"] == r else TDIM,
                        "transition": "all 0.1s",
                    },
                )
                for r in RESULTS
            ],
            style={
                "flex": "3",
                "display": "flex",
                "gap": "4px",
                "justifyContent": "center",
            },
        )

        return html.Div(
            [
                html.Div(
                    b["date"],
                    style={"flex": "0 0 54px", "fontSize": "0.78rem", "color": TDIM},
                ),
                html.Div(
                    b["match"],
                    style={"flex": "2", "fontSize": "0.78rem", "fontWeight": "600"},
                ),
                html.Div(b["selection"], style={"flex": "2", "fontSize": "0.78rem"}),
                html.Div(
                    odds_str,
                    style={
                        "flex": "0 0 54px",
                        "textAlign": "right",
                        "fontSize": "0.82rem",
                        "fontWeight": "700",
                        "fontFamily": BARLOW,
                    },
                ),
                html.Div(
                    f"{b['stake']:.2f}",
                    style={
                        "flex": "0 0 60px",
                        "textAlign": "right",
                        "fontSize": "0.82rem",
                    },
                ),
                html.Div(
                    f"{b['edge']:+.1%}",
                    style={
                        "flex": "0 0 58px",
                        "textAlign": "right",
                        "fontSize": "0.78rem",
                        "fontWeight": "700",
                        "color": edge_col,
                    },
                ),
                result_btns,
                html.Div(
                    pnl_disp,
                    style={
                        "flex": "0 0 60px",
                        "textAlign": "right",
                        "fontSize": "0.82rem",
                        "fontWeight": "700",
                        "color": pnl_col,
                    },
                ),
                html.Button(
                    "×",
                    n_clicks=0,
                    id={"type": "delete-bet-btn", "index": b["id"]},
                    style={
                        "flex": "0 0 28px",
                        "background": "none",
                        "border": "none",
                        "color": TDIM,
                        "cursor": "pointer",
                        "fontSize": "1rem",
                        "padding": "0",
                        "lineHeight": "1",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "6px",
                "padding": "8px 12px",
                "borderBottom": f"1px solid {BORDER}",
            },
        )

    rows = [bet_row(b) for b in reversed(bets)]

    table = html.Div(
        [
            header,
            *rows,
        ],
        style={
            "background": BG2,
            "border": f"1px solid {BORDER}",
            "borderRadius": "12px",
            "overflow": "hidden",
        },
    )

    return summary, table


@app.callback(
    Output("pipe-output", "children"),
    Input({"type": "pipe-btn", "index": ALL}, "n_clicks"),
    Input("pipe-all-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_pipeline(*_):
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"]

    def run(script):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )

    if "pipe-all-btn" in trigger_id:
        scripts = [s for _, s, _ in PIPELINE_STAGES[:-1]]
        logs = []
        for script in scripts:
            res = run(script)
            logs.append(
                f"{'=' * 40}\n{script}\n{'=' * 40}\n{res.stdout}\n{res.stderr}".strip()
            )
            if res.returncode != 0:
                return html.Div(
                    [
                        html.Div(
                            f"❌ Failed at {script}",
                            style={
                                "color": "#EF4444",
                                "fontWeight": "600",
                                "marginBottom": "8px",
                            },
                        ),
                        html.Pre(
                            "\n\n".join(logs),
                            style={
                                "fontSize": "0.78rem",
                                "color": TDIM,
                                "whiteSpace": "pre-wrap",
                            },
                        ),
                    ]
                )
        return html.Div(
            [
                html.Div(
                    "✅ Full pipeline complete",
                    style={
                        "color": "#22C55E",
                        "fontWeight": "600",
                        "marginBottom": "8px",
                    },
                ),
                html.Pre(
                    "\n\n".join(logs),
                    style={
                        "fontSize": "0.78rem",
                        "color": TDIM,
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ]
        )

    triggered = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    try:
        info = json.loads(triggered)
        script = info["index"]
    except Exception:
        raise PreventUpdate

    res = run(script)
    ok = res.returncode == 0
    out = (res.stdout or res.stderr or "").strip()
    return html.Div(
        [
            html.Div(
                f"{'✅' if ok else '❌'}  {script}",
                style={
                    "fontWeight": "600",
                    "marginBottom": "8px",
                    "color": "#22C55E" if ok else "#EF4444",
                },
            ),
            html.Pre(
                out,
                style={
                    "fontSize": "0.78rem",
                    "color": TDIM,
                    "whiteSpace": "pre-wrap",
                    "maxHeight": "300px",
                    "overflowY": "auto",
                },
            )
            if out
            else None,
        ]
    )


def _build_xg_table_div():
    ELO_PATH = ROOT / "data" / "elo_results.csv"
    MANUAL_PATH = ROOT / "data" / "manual_xg.csv"

    if not ELO_PATH.exists():
        return html.Div(
            "elo_results.csv not found. Run Scrape Elo first.",
            style={"color": TDIM, "padding": "16px"},
        )

    elo = pd.read_csv(ELO_PATH)
    wc26 = elo[elo["game_id"].str.startswith("26_") & (elo["world_cup"] == 1)].copy()
    if wc26.empty:
        return html.Div("No WC 2026 played games found yet.", style={"color": TDIM})

    rows = []
    for _, r in wc26.sort_values("game_id").iterrows():
        for team, opp in [(r["t1"], r["t2"]), (r["t2"], r["t1"])]:
            rows.append(
                {
                    "game_id": r["game_id"],
                    "team": team,
                    "Team": WC26_TEAMS.get(team, team),
                    "Opponent": WC26_TEAMS.get(opp, opp),
                    "xg": None,
                    "xg_conceded": None,
                }
            )
    games_df = pd.DataFrame(rows)

    if MANUAL_PATH.exists():
        existing = pd.read_csv(MANUAL_PATH)
        games_df = games_df.merge(
            existing[["game_id", "team", "xg", "xg_conceded"]],
            on=["game_id", "team"],
            how="left",
            suffixes=("", "_s"),
        )
        games_df["xg"] = games_df["xg_s"].combine_first(games_df["xg"])
        games_df["xg_conceded"] = games_df["xg_conceded_s"].combine_first(
            games_df["xg_conceded"]
        )
        games_df.drop(columns=["xg_s", "xg_conceded_s"], inplace=True, errors="ignore")

    games_df = games_df[games_df["xg"].isna() | games_df["xg_conceded"].isna()]

    if games_df.empty:
        return html.Div(
            "All games have xG data — nothing left to enter.",
            style={"color": TDIM, "padding": "16px"},
        )

    return dash_table.DataTable(
        id="xg-table",
        data=games_df.to_dict("records"),
        columns=[
            {"name": "Game", "id": "game_id", "editable": False},
            {"name": "Code", "id": "team", "editable": False},
            {"name": "Team", "id": "Team", "editable": False},
            {"name": "Opponent", "id": "Opponent", "editable": False},
            {"name": "xG For", "id": "xg", "editable": True, "type": "numeric"},
            {
                "name": "xG Against",
                "id": "xg_conceded",
                "editable": True,
                "type": "numeric",
            },
        ],
        style_cell={
            "padding": "8px 10px",
            "border": f"1px solid {BORDER}",
            "fontFamily": INTER,
            "fontSize": "0.84rem",
            "background": BG2,
            "color": TEXT,
        },
        style_header={
            "backgroundColor": BG3,
            "color": TDIM,
            "fontWeight": "600",
            "textTransform": "uppercase",
            "letterSpacing": "0.07em",
            "fontSize": "0.68rem",
            "border": f"1px solid {BORDER}",
        },
        style_data={
            "backgroundColor": BG2,
            "color": TEXT,
            "border": f"1px solid {BORDER}",
        },
        style_data_conditional=[
            {
                "if": {"state": "active"},
                "backgroundColor": "rgba(47,129,247,0.1)",
                "border": f"1px solid rgba(47,129,247,0.4)",
            }
        ],
        page_size=20,
        sort_action="native",
        style_table={"borderRadius": "8px", "overflow": "hidden"},
    )


@app.callback(
    Output("xg-table-div", "children"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def xg_editor_render(tab):
    if tab != "xg-editor":
        return no_update
    return _build_xg_table_div()


@app.callback(
    Output("xg-table-div", "children", allow_duplicate=True),
    Output("xg-save-msg", "children"),
    Input("xg-save-btn", "n_clicks"),
    State("xg-table", "data"),
    prevent_initial_call=True,
)
def xg_editor_save(save_clicks, table_data):
    MANUAL_PATH = ROOT / "data" / "manual_xg.csv"
    save_msg = ""

    if table_data:
        df = pd.DataFrame(table_data)[["game_id", "team", "xg", "xg_conceded"]]
        df = df.dropna(subset=["xg", "xg_conceded"])
        if MANUAL_PATH.exists():
            old = pd.read_csv(MANUAL_PATH)
            df = pd.concat([old, df]).drop_duplicates(
                subset=["game_id", "team"], keep="last"
            )
        df.to_csv(MANUAL_PATH, index=False)
        save_msg = f"Saved {len(df)} rows → data/manual_xg.csv"

    return _build_xg_table_div(), save_msg


# ── Parlay callbacks ──────────────────────────────────────────────────────────


@app.callback(
    Output("parlay-store", "data"),
    Input("selected-game", "data"),
    prevent_initial_call=True,
)
def clear_parlay_on_game_change(_):
    return {}


@app.callback(
    Output("parlay-store", "data", allow_duplicate=True),
    Input({"type": "parlay-btn", "index": ALL}, "n_clicks"),
    State("parlay-store", "data"),
    prevent_initial_call=True,
)
def toggle_parlay_leg(btn_clicks, current):
    ctx = callback_context
    if not ctx.triggered or not any(c for c in btn_clicks if c):
        raise PreventUpdate

    trigger = ctx.triggered[0]["prop_id"]
    try:
        info = json.loads(trigger.split(".")[0])
        gid, outcome = info["index"].rsplit("__", 1)
    except Exception:
        raise PreventUpdate

    sels = dict(current or {})
    if sels.get(gid) == outcome:
        sels.pop(gid, None)
    else:
        sels[gid] = outcome
    return sels


def _parlay_prob(r, outcome):
    return {"w": float(r["p_win"]), "d": float(r["p_draw"]), "l": float(r["p_loss"])}[
        outcome
    ]


def _parlay_label(r, outcome):
    t1n, t2n = full(r["team"]), full(r["opponent"])
    return {"w": f"{t1n} Win", "d": "Draw", "l": f"{t2n} Win"}[outcome]


@app.callback(
    Output("parlay-legs-div", "children"),
    Output("parlay-summary-div", "children"),
    Output("parlay-odds-input-row", "style"),
    Input("modal-tabs", "value"),
    Input("parlay-store", "data"),
    State("selected-game", "data"),
    State("game-df-store", "data"),
)
def render_parlay(tab, sels, game_id, store_json):
    empty = html.Div(), html.Div(), {"display": "none"}
    if tab != "parlay" or not game_id or not store_json:
        return empty

    gdf = pd.read_json(StringIO(store_json), orient="split")
    row = gdf[gdf["game_id"] == game_id]
    if row.empty:
        return empty
    game_date = parse_date(game_id)

    gdf["_date"] = pd.to_datetime(gdf["date_iso"]).dt.date
    same_day = gdf[gdf["_date"] == game_date].reset_index(drop=True)

    sels = sels or {}

    # ── Leg cards ────────────────────────────────────────────────────────────
    cards = []
    for _, r in same_day.iterrows():
        gid = r["game_id"]
        t1n = full(r["team"])
        t2n = full(r["opponent"])
        sel = sels.get(gid)
        tc = TEAM_COLORS.get(r["team"], "#444")
        oc = TEAM_COLORS.get(r["opponent"], "#444")

        def _btn(outcome, label, prob, game_id=gid, selected=sel):
            is_sel = selected == outcome
            return html.Button(
                f"{label}  {prob * 100:.0f}%",
                id={"type": "parlay-btn", "index": f"{game_id}__{outcome}"},
                n_clicks=0,
                style={
                    "padding": "5px 11px",
                    "fontSize": "0.75rem",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "border": f"1px solid {'#2563EB' if is_sel else BORDER}",
                    "background": "#2563EB" if is_sel else BG,
                    "color": "#fff" if is_sel else TDIM,
                    "fontWeight": "700" if is_sel else "500",
                    "transition": "all 0.12s",
                },
            )

        btns = [
            _btn("w", f"{t1n} Win", r["p_win"]),
            _btn("d", "Draw", r["p_draw"]),
            _btn("l", f"{t2n} Win", r["p_loss"]),
        ]

        header_color = (
            tc
            if sel == "w"
            else (oc if sel == "l" else (ACCENT if sel == "d" else BORDER))
        )

        cards.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                r["team"],
                                style={
                                    "background": tc,
                                    "color": _fg(tc),
                                    "padding": "2px 7px",
                                    "borderRadius": "3px",
                                    "fontFamily": BARLOW,
                                    "fontSize": "0.9rem",
                                    "fontWeight": "800",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Span(
                                "vs",
                                style={
                                    "color": TDIM,
                                    "fontSize": "0.75rem",
                                    "marginRight": "6px",
                                },
                            ),
                            html.Span(
                                r["opponent"],
                                style={
                                    "background": oc,
                                    "color": _fg(oc),
                                    "padding": "2px 7px",
                                    "borderRadius": "3px",
                                    "fontFamily": BARLOW,
                                    "fontSize": "0.9rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "marginBottom": "6px",
                            "display": "flex",
                            "alignItems": "center",
                        },
                    ),
                    html.Div(
                        btns,
                        style={"display": "flex", "gap": "6px", "flexWrap": "wrap"},
                    ),
                ],
                style={
                    "padding": "10px 12px",
                    "marginBottom": "6px",
                    "borderRadius": "8px",
                    "background": BG,
                    "border": f"1px solid {header_color}",
                },
            )
        )

    legs_div = html.Div(cards)

    # ── Summary ───────────────────────────────────────────────────────────────
    selected_legs = [(gid, o) for gid, o in sels.items()]
    n_legs = len(selected_legs)

    if n_legs == 0:
        summary = html.Div(
            "Click an outcome on each game to build your parlay.",
            style={"fontSize": "0.82rem", "color": TDIM},
        )
        return legs_div, summary, {"display": "none"}

    if n_legs == 1:
        summary = html.Div(
            "Add at least one more leg to calculate parlay odds.",
            style={"fontSize": "0.82rem", "color": TDIM},
        )
        return legs_div, summary, {"display": "none"}

    # Combined probability
    combined_prob = 1.0
    leg_labels = []
    for gid, outcome in selected_legs:
        gr = same_day[same_day["game_id"] == gid]
        if gr.empty:
            continue
        r = gr.iloc[0]
        combined_prob *= _parlay_prob(r, outcome)
        leg_labels.append(_parlay_label(r, outcome))

    p = combined_prob
    if p >= 0.5:
        implied_odds = f"-{int(round(p / (1 - p) * 100))}"
    else:
        implied_odds = f"+{int(round((1 - p) / p * 100))}"

    legs_str = "  +  ".join(leg_labels)

    summary = html.Div(
        [
            html.Div(
                legs_str,
                style={
                    "fontSize": "0.78rem",
                    "color": TEXT,
                    "fontWeight": "600",
                    "marginBottom": "8px",
                    "lineHeight": "1.5",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                f"{n_legs}-leg parlay",
                                style={
                                    "fontSize": "0.68rem",
                                    "color": TDIM,
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                },
                            ),
                            html.Div(
                                f"{p * 100:.2f}%",
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "1.4rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "background": BG3,
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "flex": "1",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Model implied",
                                style={
                                    "fontSize": "0.68rem",
                                    "color": TDIM,
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                },
                            ),
                            html.Div(
                                implied_odds,
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "1.4rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "background": BG3,
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "flex": "1",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
        ]
    )

    return legs_div, summary, {"display": "block", "marginTop": "12px"}


@app.callback(
    Output("parlay-edge-display", "children"),
    Input("parlay-market-odds", "value"),
    State("parlay-store", "data"),
    State("game-df-store", "data"),
    State("selected-game", "data"),
    prevent_initial_call=True,
)
def update_parlay_edge(market_odds, sels, store_json, game_id):
    if not sels or not store_json or not game_id:
        return html.Div()

    sels = sels or {}
    if len(sels) < 2:
        return html.Div()

    gdf = pd.read_json(StringIO(store_json), orient="split")
    game_date = parse_date(game_id)
    gdf["_date"] = pd.to_datetime(gdf["date_iso"]).dt.date
    same_day = gdf[gdf["_date"] == game_date]

    combined_prob = 1.0
    for gid, outcome in sels.items():
        gr = same_day[same_day["game_id"] == gid]
        if gr.empty:
            continue
        combined_prob *= _parlay_prob(gr.iloc[0], outcome)

    try:
        mo = int(market_odds)
    except (TypeError, ValueError):
        return html.Div()

    market_impl = abs(mo) / (abs(mo) + 100) if mo < 0 else 100 / (mo + 100)
    edge = combined_prob - market_impl

    if mo > 0:
        payout_per_unit = mo / 100
    else:
        payout_per_unit = 100 / abs(mo)

    kelly_f = (payout_per_unit * combined_prob - (1 - combined_prob)) / payout_per_unit
    kelly_f = max(kelly_f, 0)

    edge_pct = edge * 100
    edge_color = "#22C55E" if edge > 0.02 else ("#F59E0B" if edge > 0 else "#EF4444")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Market implied",
                                style={
                                    "fontSize": "0.68rem",
                                    "color": TDIM,
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                },
                            ),
                            html.Div(
                                f"{market_impl * 100:.2f}%",
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "1.3rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "background": BG3,
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "flex": "1",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Edge",
                                style={
                                    "fontSize": "0.68rem",
                                    "color": TDIM,
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                },
                            ),
                            html.Div(
                                f"{edge_pct:+.2f}%",
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "1.3rem",
                                    "fontWeight": "800",
                                    "color": edge_color,
                                },
                            ),
                        ],
                        style={
                            "background": BG3,
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "flex": "1",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Kelly %",
                                style={
                                    "fontSize": "0.68rem",
                                    "color": TDIM,
                                    "fontWeight": "700",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "0.06em",
                                },
                            ),
                            html.Div(
                                f"{kelly_f * 100:.1f}%",
                                style={
                                    "fontFamily": BARLOW,
                                    "fontSize": "1.3rem",
                                    "fontWeight": "800",
                                },
                            ),
                        ],
                        style={
                            "background": BG3,
                            "borderRadius": "8px",
                            "padding": "10px 14px",
                            "flex": "1",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
        ]
    )


@app.callback(
    Output("history-game-df-store", "data"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def load_history_game_store(tab):
    if tab != "history":
        return no_update
    games, _ = load_history_data()
    if games is None or games.empty:
        return None
    games_out = games.drop(columns=["_date"], errors="ignore")
    return games_out.to_json(orient="split", default_handler=str)


@app.callback(
    Output("history-date-dropdown", "options"),
    Input("history-game-df-store", "data"),
)
def populate_history_dates(store_json):
    if not store_json:
        return []
    games = pd.read_json(StringIO(store_json), orient="split")
    dates = sorted(pd.to_datetime(games["date_iso"]).dt.date.unique(), reverse=True)
    return [{"label": d.strftime("%a, %b %d"), "value": d.isoformat()} for d in dates]


@app.callback(
    Output("history-date-dropdown", "value"),
    Input("history-date-mode", "value"),
    prevent_initial_call=True,
)
def clear_history_dropdown(_):
    return None


@app.callback(
    Output("history-content", "children"),
    Output("history-game-count", "children"),
    Input("main-tabs", "value"),
    Input("history-date-mode", "value"),
    Input("history-date-dropdown", "value"),
    Input("history-game-df-store", "data"),
)
def render_history(tab, date_mode, dropdown_date, store_json):
    empty = no_update, ""
    if tab != "history":
        return empty

    if not store_json:
        msg = html.Div(
            [
                html.Div("No past WC2026 games found.", style={"color": TDIM}),
                html.Div(
                    "Run Scrape Elo to fetch results, then Retro Predictions to generate model predictions.",
                    style={"fontSize": "0.83rem", "color": TDIM, "marginTop": "6px"},
                ),
            ]
        )
        return msg, ""

    games = pd.read_json(StringIO(store_json), orient="split")
    games["_date"] = pd.to_datetime(games["date_iso"]).dt.date

    wp = games[games["has_preds"].astype(bool)]
    metrics = {
        "total_games": len(games),
        "with_preds": len(wp),
        "correct": int(wp["is_correct"].sum()) if len(wp) else 0,
        "accuracy": float(wp["is_correct"].mean()) if len(wp) else 0.0,
        "avg_rps": float(wp["rps_score"].mean()) if len(wp) else 0.0,
    }

    # Date filter — applies to cards only; summary metrics are always all-time
    if dropdown_date:
        try:
            pick = datetime.date.fromisoformat(dropdown_date)
            filtered = games[games["_date"] == pick]
        except Exception:
            filtered = games
    else:
        filtered = games

    total = metrics["total_games"]
    with_preds = metrics["with_preds"]
    correct = metrics["correct"]
    accuracy = metrics["accuracy"]
    avg_rps = metrics["avg_rps"]

    summary = html.Div(
        [
            _metric_card(str(total), "Games Played"),
            _metric_card(
                f"{correct}/{with_preds}" if with_preds else "—", "Correct Picks"
            ),
            _metric_card(f"{accuracy * 100:.0f}%" if with_preds else "—", "Accuracy"),
            _metric_card(f"{avg_rps:.3f}" if with_preds else "—", "Avg RPS"),
        ],
        style={
            "display": "flex",
            "gap": "10px",
            "marginBottom": "16px",
            "flexWrap": "wrap",
        },
    )

    warn = None
    if with_preds == 0:
        warn = html.Div(
            "⚠  No model predictions yet. Run Retro Predictions in the Pipeline tab.",
            style={
                "background": "rgba(234,179,8,0.1)",
                "border": "1px solid rgba(234,179,8,0.3)",
                "borderRadius": "8px",
                "padding": "10px 14px",
                "fontSize": "0.83rem",
                "color": "#A16207",
                "marginBottom": "12px",
            },
        )

    n = len(filtered)
    count_str = f"{n} match{'es' if n != 1 else ''}"

    if filtered.empty:
        cards_div = html.Div(
            f"No games for {dropdown_date}.",
            style={"color": TDIM, "padding": "8px"},
        )
    else:
        cards_div = html.Div([history_match_card(r) for _, r in filtered.iterrows()])

    content = html.Div([summary, warn, cards_div] if warn else [summary, cards_div])
    return content, count_str


if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", "8050"))
    debug = os.environ.get("DASH_DEBUG", "0") == "1"
    app.run(debug=debug, port=port)
