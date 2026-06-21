#!/usr/bin/env Rscript

# Transfermarkt WC 2026 national team squad scraper.
# Discovers each national team's Transfermarkt verein ID via search,
# then scrapes the squad (kader) page directly with rvest.
# (worldfootballR's tm_squad_stats() breaks on national team HTML — different structure.)
#
# Usage:
#   Rscript scripts/tm_squads.R            # all 48 WC 2026 teams
#   Rscript scripts/tm_squads.R --test     # England only
#
# Output:     data/tm_squads.csv


suppressPackageStartupMessages({
  for (pkg in c("httr", "rvest", "dplyr")) {
    if (!requireNamespace(pkg, quietly = TRUE))
      install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
})

OUTFILE   <- "./data/tm_squads.csv"
SLEEP_SEC <- 4
BASE_URL   <- "https://www.transfermarkt.com"
TM_HEADERS <- httr::add_headers(
  "User-Agent"      = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
  "Accept-Language" = "en-US,en;q=0.9"
)

args        <- commandArgs(trailingOnly = TRUE)
test_mode   <- "--test" %in% args
append_mode <- "--append" %in% args

# --teams CW,CD,CZ  — run only these abbrs
teams_arg  <- grep("^--teams=", args, value = TRUE)
only_abbrs <- if (length(teams_arg) > 0) {
  strsplit(sub("^--teams=", "", teams_arg[1]), ",")[[1]]
} else {
  character(0)
}

# ── WC 2026 teams (full names used for TM search) ────────────────────────────
WC2026_TEAMS <- list(
  # CONCACAF
  list(abbr="US",  name="United States"),
  list(abbr="MX",  name="Mexico"),
  list(abbr="CA",  name="Canada"),
  list(abbr="CW",  name="Curacao"),
  list(abbr="HT",  name="Haiti"),
  list(abbr="PA",  name="Panama"),
  # CONMEBOL
  list(abbr="AR",  name="Argentina"),
  list(abbr="BR",  name="Brazil"),
  list(abbr="CO",  name="Colombia"),
  list(abbr="EC",  name="Ecuador"),
  list(abbr="PY",  name="Paraguay"),
  list(abbr="UY",  name="Uruguay"),
  # UEFA
  list(abbr="AT",  name="Austria"),
  list(abbr="BE",  name="Belgium"),
  list(abbr="BA",  name="Bosnia-Herzegovina"),
  list(abbr="HR",  name="Croatia"),
  list(abbr="CZ",  name="Czech Republic"),
  list(abbr="EN",  name="England"),
  list(abbr="FR",  name="France"),
  list(abbr="DE",  name="Germany"),
  list(abbr="NL",  name="Netherlands"),
  list(abbr="NO",  name="Norway"),
  list(abbr="PT",  name="Portugal"),
  list(abbr="SC",  name="Scotland"),
  list(abbr="ES",  name="Spain"),
  list(abbr="SE",  name="Sweden"),
  list(abbr="CH",  name="Switzerland"),
  list(abbr="TR",  name="Turkey"),
  # CAF
  list(abbr="DZ",  name="Algeria"),
  list(abbr="CV",  name="Cape Verde"),
  list(abbr="CD",  name="DR Congo"),
  list(abbr="CI",  name="Ivory Coast"),
  list(abbr="EG",  name="Egypt"),
  list(abbr="GH",  name="Ghana"),
  list(abbr="MA",  name="Morocco"),
  list(abbr="SN",  name="Senegal"),
  list(abbr="ZA",  name="South Africa"),
  list(abbr="TN",  name="Tunisia"),
  # AFC
  list(abbr="AU",  name="Australia"),
  list(abbr="IQ",  name="Iraq"),
  list(abbr="IR",  name="Iran"),
  list(abbr="JP",  name="Japan"),
  list(abbr="JO",  name="Jordan"),
  list(abbr="QA",  name="Qatar"),
  list(abbr="SA",  name="Saudi Arabia"),
  list(abbr="KR",  name="South Korea"),
  list(abbr="UZ",  name="Uzbekistan"),
  # OFC
  list(abbr="NZ",  name="New Zealand")
)

# ── Known Transfermarkt kader URLs (verified) ─────────────────────────────────
# Format: team abbr -> full kader URL
# Add more as you verify them at transfermarkt.com
KNOWN_URLS <- list(
  EN = "https://www.transfermarkt.com/england/kader/verein/3299",
  DE = "https://www.transfermarkt.com/deutschland/kader/verein/3262",
  FR = "https://www.transfermarkt.com/frankreich/kader/verein/3377",
  ES = "https://www.transfermarkt.com/spanien/kader/verein/3375",
  PT = "https://www.transfermarkt.com/portugal/kader/verein/3300",
  US = "https://www.transfermarkt.com/vereinigte-staaten/kader/verein/3505",
  TR = "https://www.transfermarkt.com/turkei/kader/verein/3381s",
  NL = "https://www.transfermarkt.com/niederlande/kader/verein/3379",
  BE = "https://www.transfermarkt.com/belgien/kader/verein/3382",
  BR = "https://www.transfermarkt.com/brasilien/kader/verein/3439",
  CW = "https://www.transfermarkt.com/curacao/kader/verein/32364",
  CD = "https://www.transfermarkt.com/demokratische-republik-kongo/kader/verein/3854",
  CZ = "https://www.transfermarkt.com/tschechien/kader/verein/3445"
)

# ── Find verein ID for a national team via TM search ─────────────────────────
find_verein_id <- function(team_abbr, team_name) {
  # Use known URL if available
  if (team_abbr %in% names(KNOWN_URLS)) {
    return(KNOWN_URLS[[team_abbr]])
  }

  search_url <- sprintf(
    "%s/schnellsuche/ergebnis/schnellsuche?query=%s&Verein=Verein",
    BASE_URL, utils::URLencode(team_name)
  )
  resp <- tryCatch(
    httr::GET(search_url, TM_HEADERS),
    error = function(e) NULL
  )
  if (is.null(resp) || httr::status_code(resp) != 200) return(NULL)

  page  <- httr::content(resp, as = "text", encoding = "UTF-8") |> rvest::read_html()
  hrefs <- page |> rvest::html_nodes("a") |> rvest::html_attr("href")
  texts <- page |> rvest::html_nodes("a") |> rvest::html_text(trim = TRUE)

  # Match links where the anchor text exactly matches the team name
  exact_match_idx <- which(tolower(texts) == tolower(team_name) &
                           grepl("/startseite/verein/\\d+", hrefs))
  if (length(exact_match_idx) > 0) {
    href <- hrefs[exact_match_idx[1]]
    verein_id <- gsub(".*/verein/(\\d+).*", "\\1", href)
    slug_tm   <- gsub("^/([^/]+)/.*", "\\1", href)
    return(sprintf("%s/%s/kader/verein/%s", BASE_URL, slug_tm, verein_id))
  }

  # Fallback: slug-based match (e.g. "england" -> /england/startseite/verein/...)
  slug    <- tolower(gsub("[^a-z0-9]", "-", gsub("\\s+", "-", team_name)))
  pattern <- sprintf("^/%s/startseite/verein/\\d+", slug)
  matches <- unique(hrefs[!is.na(hrefs) & grepl(pattern, hrefs)])
  if (length(matches) == 0) return(NULL)

  verein_id <- gsub(".*/verein/(\\d+).*", "\\1", matches[1])
  slug_tm   <- gsub("^/([^/]+)/.*", "\\1", matches[1])
  sprintf("%s/%s/kader/verein/%s", BASE_URL, slug_tm, verein_id)
}

# ── Scrape squad from a national team kader page ─────────────────────────────
scrape_squad <- function(kader_url, team_abbr, team_name) {
  resp <- tryCatch(
    httr::GET(kader_url, TM_HEADERS),
    error = function(e) NULL
  )
  if (is.null(resp) || httr::status_code(resp) != 200) {
    message(sprintf("    HTTP %s", if (is.null(resp)) "error" else httr::status_code(resp)))
    return(NULL)
  }

  page <- httr::content(resp, as = "text", encoding = "UTF-8") |> rvest::read_html()

  # Player rows have exactly 8 tds
  all_rows    <- page |> rvest::html_nodes("table.items tbody tr")
  player_rows <- Filter(function(r) length(rvest::html_nodes(r, "td")) == 8, all_rows)

  if (length(player_rows) == 0) {
    message("    No player rows found.")
    return(NULL)
  }

  extract_td <- function(row, n, attr = NULL) {
    td <- rvest::html_nodes(row, "td")[[n]]
    if (is.null(attr)) rvest::html_text(td, trim = TRUE)
    else rvest::html_attr(td, attr)
  }

  rows_data <- lapply(player_rows, function(r) {
    tds <- rvest::html_nodes(r, "td")

    # shirt number
    shirt <- rvest::html_text(tds[[1]], trim = TRUE)

    # player name + URL from td[4] hauptlink
    name_node <- rvest::html_node(tds[[4]], "a")
    player_name <- if (!is.null(name_node)) rvest::html_text(name_node, trim = TRUE) else ""
    player_href <- if (!is.null(name_node)) rvest::html_attr(name_node, "href") else NA
    player_url  <- if (!is.na(player_href)) paste0(BASE_URL, player_href) else NA

    # nationality flag alt text from td[4]
    nat_node    <- rvest::html_node(tds[[4]], "img.flaggenrahmen")
    nationality <- if (!is.null(nat_node)) rvest::html_attr(nat_node, "title") else NA

    # position td[5], age td[6], market value td[8]
    position     <- rvest::html_text(tds[[5]], trim = TRUE)
    age          <- rvest::html_text(tds[[6]], trim = TRUE)
    market_value <- rvest::html_text(tds[[8]], trim = TRUE)

    data.frame(
      team_abbr    = team_abbr,
      team_name    = team_name,
      shirt_number = shirt,
      player_name  = player_name,
      player_url   = player_url,
      nationality  = nationality,
      position     = position,
      age          = age,
      market_value = market_value,
      kader_url    = kader_url,
      stringsAsFactors = FALSE
    )
  })

  dplyr::bind_rows(rows_data)
}

# ── Select teams to process ───────────────────────────────────────────────────
teams <- if (test_mode) {
  message("\n=== TEST MODE: England only ===")
  list(list(abbr = "EN", name = "England"))
} else if (length(only_abbrs) > 0) {
  message(sprintf("\n=== TARGETED MODE: %s ===", paste(only_abbrs, collapse = ", ")))
  Filter(function(t) t$abbr %in% only_abbrs, WC2026_TEAMS)
} else {
  WC2026_TEAMS
}

# ── Load existing output for append mode ─────────────────────────────────────
done_abbrs <- character(0)
parts      <- list()

if (append_mode && file.exists(OUTFILE)) {
  existing   <- read.csv(OUTFILE, stringsAsFactors = FALSE, colClasses = "character")
  # Remove rows for teams we're about to re-scrape so we don't duplicate
  existing   <- existing[!(existing$team_abbr %in% sapply(teams, `[[`, "abbr")), ]
  parts      <- list(existing)
  done_abbrs <- character(0)   # always re-scrape the targeted teams
  message(sprintf("Append mode: keeping %d existing rows, re-scraping %s.",
                  nrow(existing), paste(sapply(teams, `[[`, "abbr"), collapse = ", ")))
}

# ── Main loop ─────────────────────────────────────────────────────────────────
message(sprintf("\nProcessing %d team(s)...", length(teams)))

for (team in teams) {
  if (team$abbr %in% done_abbrs) next
  message(sprintf("\n  [%s] %s", team$abbr, team$name))

  # Step 1: find kader URL
  kader_url <- find_verein_id(team$abbr, team$name)
  Sys.sleep(SLEEP_SEC)

  if (is.null(kader_url)) {
    message("    Could not find Transfermarkt page — skipping.")
    next
  }
  message(sprintf("    %s", kader_url))

  # Step 2: scrape squad
  squad <- scrape_squad(kader_url, team$abbr, team$name)
  Sys.sleep(SLEEP_SEC)

  if (!is.null(squad) && nrow(squad) > 0) {
    parts[[length(parts) + 1]] <- squad
    done_abbrs <- c(done_abbrs, team$abbr)
    message(sprintf("    %d players", nrow(squad)))
  }
}

# ── Write output ──────────────────────────────────────────────────────────────
if (length(parts) == 0) {
  message("\nNo data collected.")
  write.csv(data.frame(), OUTFILE, row.names = FALSE)
} else {
  final <- dplyr::bind_rows(parts)
  write.csv(final, OUTFILE, row.names = FALSE)
  message(sprintf("\nDone. %d players across %d teams written to %s",
                  nrow(final), length(unique(final$team_name)), OUTFILE))
}
