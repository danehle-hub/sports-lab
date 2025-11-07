TEAM_NAME_TO_CODE = {
    "ARIZONA CARDINALS": "ARI",
    "ATLANTA FALCONS": "ATL",
    "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF",
    "CAROLINA PANTHERS": "CAR",
    "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN",
    "CLEVELAND BROWNS": "CLE",
    "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN",
    "DETROIT LIONS": "DET",
    "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU",
    "INDIANAPOLIS COLTS": "IND",
    "JACKSONVILLE JAGUARS": "JAX",
    "KANSAS CITY CHIEFS": "KC",
    "LAS VEGAS RAIDERS": "LV",
    "OAKLAND RAIDERS": "OAK",  # pre-relocation games
    "LOS ANGELES CHARGERS": "LAC",
    "SAN DIEGO CHARGERS": "SD", # historical mapping
    "LOS ANGELES RAMS": "LAR",
    "ST. LOUIS RAMS": "STL", # historical mapping
    "MIAMI DOLPHINS": "MIA",
    "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE",
    "NEW ORLEANS SAINTS": "NO",
    "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ",
    "PHILADELPHIA EAGLES": "PHI",
    "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF",
    "SEATTLE SEAHAWKS": "SEA",
    "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN",
    "WASHINGTON COMMANDERS": "WAS",
    "WASHINGTON FOOTBALL TEAM": "WAS",
    "WASHINGTON REDSKINS": "WAS"
}
# --- Add these lines at the bottom of models/team_map.py ---

# Identity mappings so codes map to themselves
CODE_IDENTITY = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC",
    "LAC":"LAC","LAR":"LAR","LV":"LV","MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SEA":"SEA","SF":"SF","TB":"TB","TEN":"TEN","WAS":"WAS"
}
TEAM_NAME_TO_CODE.update(CODE_IDENTITY)

# Helpful aliases for legacy/ambiguous names
TEAM_NAME_TO_CODE.update({
    "LA":"LAR",
    "LA RAMS":"LAR",
    "LOS ANGELES RAMS":"LAR",
    "ST. LOUIS RAMS":"LAR",

    "SAN DIEGO CHARGERS":"LAC",
    "SD":"LAC",

    "OAKLAND RAIDERS":"LV",
    "OAK":"LV",
    "LAS VEGAS RAIDERS":"LV",
    "LVR":"LV",

    "WASHINGTON":"WAS",
    "WASHINGTON REDSKINS":"WAS",
    "WASHINGTON FOOTBALL TEAM":"WAS",
    "WSH":"WAS",

    "TAMPA BAY":"TB",
    "NEW ENGLAND":"NE",
    "NEW ORLEANS":"NO",
    "SAN FRANCISCO":"SF",
    "GREEN BAY":"GB",
})
