WC26_TEAMS = {
    # CONCACAF
    "US": "United States",
    "MX": "Mexico",
    "CA": "Canada",
    "CW": "Curaçao",
    "HT": "Haiti",
    "PA": "Panama",
    # CONMEBOL
    "AR": "Argentina",
    "BR": "Brazil",
    "CO": "Colombia",
    "EC": "Ecuador",
    "PY": "Paraguay",
    "UY": "Uruguay",
    # UEFA
    "AT": "Austria",
    "BE": "Belgium",
    "BA": "Bosnia and Herzegovina",
    "HR": "Croatia",
    "CZ": "Czechia",
    "EN": "England",
    "FR": "France",
    "DE": "Germany",
    "NL": "Netherlands",
    "NO": "Norway",
    "PT": "Portugal",
    "SQ": "Scotland",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "TR": "Turkey",
    # CAF
    "DZ": "Algeria",
    "CV": "Cabo Verde",
    "CD": "DR Congo",
    "CI": "Côte d'Ivoire",
    "EG": "Egypt",
    "GH": "Ghana",
    "MA": "Morocco",
    "SN": "Senegal",
    "ZA": "South Africa",
    "TN": "Tunisia",
    # AFC
    "AU": "Australia",
    "IQ": "Iraq",
    "IR": "Iran",
    "JP": "Japan",
    "JO": "Jordan",
    "QA": "Qatar",
    "SA": "Saudi Arabia",
    "KR": "South Korea",
    "UZ": "Uzbekistan",
    # OFC
    "NZ": "New Zealand",
}

WC26_ABBR = {v: k for k, v in WC26_TEAMS.items()}

COMP_MAP = {
    # World Cup
    "WC": "wc",
    # Continental Championships
    "EC": "cont",  # UEFA Euros
    "CA": "cont",  # Copa América
    "AR": "cont",  # Africa Cup of Nations
    "AC": "cont",  # AFC Asian Cup
    "CCH": "cont",  # CONCACAF Gold Cup
    "OC": "cont",  # OFC Nations Cup
    "CC": "cont",  # FIFA Confederations Cup
    "IC": "cont",  # Finalissima
    # WC Qualifiers
    "WQ": "wc_qualifier",  # UEFA WCQ
    "WQA": "wc_qualifier",  # CAF WCQ
    "WQS": "wc_qualifier",  # CONMEBOL WCQ
    "WQO": "wc_qualifier",  # OFC WCQ
    "EQ": "wc_qualifier",  # UEFA Euro qualifier
    "FQ": "wc_qualifier",  # CAF WCQ (alternate)
    "SQ": "wc_qualifier",  # AFC WCQ
    "NQU": "wc_qualifier",  # CONCACAF WCQ (UNCAF zone)
    "NQC": "wc_qualifier",  # CONCACAF WCQ (Caribbean zone)
    "CFC": "wc_qualifier",  # CONCACAF WCQ final round
    "EAQ": "wc_qualifier",  # EAFF WCQ
    "AOC": "wc_qualifier",  # AFC/OFC intercontinental playoff
    "OSN": "wc_qualifier",  # intercontinental playoff
    "FFS": "wc_qualifier",  # CAF WCQ (sub-group)
    "CRO": "wc_qualifier",  # CAF WCQ / CHAN qualifier
    # Continental Qualifiers (non-WC)
    "CCQ": "cont_quali",  # CONCACAF Gold Cup qualifier
    "FCQ": "cont_quali",  # CFU qualifying
    "FQC": "cont_quali",  # CFU qualifying
    "FQB": "cont_quali",  # CFU qualifying
    "FQP": "cont_quali",  # CFU qualifying
    "CLQ": "cont_quali",  # CONCACAF League qualifier
    "CLA": "cont_quali",  # CONCACAF qualifier
    "CLB": "cont_quali",  # CFU qualifying
    "CAQ": "cont_quali",  # Copa América qualifying
    "FBQ": "cont_quali",  # CFU qualifying
    "CRC": "cont_quali",  # Caribbean qualifying
    "CBC": "cont_quali",  # Caribbean qualifying
    # Nations Leagues
    "ENA": "nations_league",  # UEFA NL A
    "ENB": "nations_league",  # UEFA NL B
    "ENC": "nations_league",  # UEFA NL C
    "ENL": "nations_league",  # UEFA NL Finals
    "EAB": "nations_league",  # UEFA NL B playoffs
    "CNL": "nations_league",  # CONCACAF Nations League
    # Sub-continental / Invitationals
    "F": "friendly",
    "BAL": "friendly",
    "RPB": "friendly",
    "CBG": "sub_cont",  # China/HK invitational
    "MJT": "sub_cont",  # Croatia invitational
    "KNG": "sub_cont",  # King's Cup (Thailand)
    "KRN": "sub_cont",  # Kirin Cup (Japan)
    "LGC": "sub_cont",  # multi-team invitational
    "FT": "sub_cont",  # WAFU/CAF sub-regional
    "GLF": "sub_cont",  # Gulf Cup
    "CSF": "sub_cont",  # COSAFA Cup
    "WAH": "sub_cont",  # West Asian Championship
    "WAG": "sub_cont",  # West Asian Games
    "ARC": "sub_cont",  # Arab Cup
    "EAH": "sub_cont",  # EAFF Championship
    "IOG": "sub_cont",  # Indian Ocean Games
    "PMC": "sub_cont",  # Arabian Peninsula Championship
    "TGC": "sub_cont",  # Tiger Cup / AFF
    "NMC": "sub_cont",  # Nelson Mandela Challenge
    "IND": "sub_cont",  # COSAFA Independence Cup
    "UNC": "sub_cont",  # UNCAF / Central American Cup
    "ILG": "sub_cont",  # West Asia invitational
    "FRC": "sub_cont",  # AFC Challenge Cup / West Asia
    "FCC": "sub_cont",  # China invitational
    "ABS": "sub_cont",  # ABC Tournament (Dutch Caribbean)
    "ABC": "sub_cont",  # ABC Tournament
    "CEC": "sub_cont",  # CECAFA Cup
    "NBT": "sub_cont",  # Nile Basin Tournament
    "NLC": "sub_cont",  # Argentina-Brazil Superclásico series
    "ATV": "sub_cont",  # Central Asia tournament
    "INC": "sub_cont",  # West Asia invitational
    "CHN": "sub_cont",  # China Cup
    "ADI": "sub_cont",  # Argentina-Mexico bilateral
    "FRT": "sub_cont",  # West Asian invitational
    "NRZ": "sub_cont",  # Nowruz Cup
    "CNU": "sub_cont",  # CAFA Nations Cup
    "CDS": "sub_cont",  # intercontinental invitational
    "ASH": "sub_cont",  # Australia-NZ bilateral
    "FFC": "sub_cont",  # Egypt Cup invitational
    "UTC": "sub_cont",  # WAFU Cup
    "NSM": "sub_cont",  # WC prep tournament
    "BGC": "sub_cont",  # Bengal Gold Cup
    "PMT": "sub_cont",  # South Asian tournament
    "INL": "sub_cont",  # intercontinental invitational
    "ALM": "sub_cont",  # Lusophone invitational
    "NTC": "sub_cont",  # Home Nations
    "AAC": "sub_cont",  # Afro-Asian Cup
    "PAR": "sub_cont",  # Arab Nations Cup
    "ACC": "sub_cont",  # Amílcar Cabral Cup
    "FCS": "sub_cont",  # Italy-Argentina bilateral
    "CHC": "sub_cont",  # Copa Centenario / bilateral
    "CYC": "sub_cont",  # bilateral
    "BSE": "sub_cont",  # bilateral
    "TOI": "sub_cont",  # bilateral
    "ANT": "sub_cont",  # bilateral
    "CPC": "sub_cont",  # Copa del Pacifico (Bolivia/Paraguay)
    "CTS": "sub_cont",  # Austria invitational
    "DCS": "sub_cont",  # Caribbean bilateral series
}

POS_GROUP_MAP = {
    "Centre-Back": "defense",
    "Left-Back": "defense",
    "Right-Back": "defense",
    "Defensive Midfield": "midfield",
    "Central Midfield": "midfield",
    "Attacking Midfield": "midfield",
    "Right Midfield": "midfield",
    "Left Midfield": "midfield",
    "Centre-Forward": "attack",
    "Left Winger": "attack",
    "Right Winger": "attack",
    "Second Striker": "attack",
    "Goalkeeper": "goal",
}

FS_TO_CANONICAL = {
    "Cape Verde Islands": "Cabo Verde",
    "Congo DR": "DR Congo",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Korea Republic": "South Korea",
    "Türkiye": "Turkey",
    "USMNT": "United States",
}

CANONICAL_TO_FS = {v: k for k, v in FS_TO_CANONICAL.items()}

TM_TO_CANONICAL = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde": "Cabo Verde",
    "Curacao": "Curaçao",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
}

CANONICAL_TO_TM = {v: k for k, v in TM_TO_CANONICAL.items()}
