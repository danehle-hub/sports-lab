import pandas as pd
from pathlib import Path

raw_path = Path('data/raw/nfl_games.csv')
out_path = Path('data/clean/games_clean.csv')

print('Using raw file:', raw_path.resolve())
if not raw_path.exists():
    raise FileNotFoundError(f'Cannot find {raw_path}. Put your CSV there.')

df = pd.read_csv(raw_path)
print('Loaded rows:', len(df))

# Normalize columns
df.columns = [c.lower().strip() for c in df.columns]

# Required columns sanity check
required = [
    'schedule_date','team_home','team_away','score_home','score_away',
    'team_favorite_id','spread_favorite'
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f'Missing required columns: {missing}')

# Convert date
df['schedule_date'] = pd.to_datetime(df['schedule_date'], errors='coerce')

# Compute margin
df['home_margin'] = df['score_home'].astype(float) - df['score_away'].astype(float)

# Winner
df['winner'] = df['home_margin'].apply(lambda x: 'home' if x > 0 else ('away' if x < 0 else 'push'))

# Spread numeric
df['spread_favorite'] = pd.to_numeric(df['spread_favorite'], errors='coerce')

# Home is favorite?
df['home_is_favorite'] = df['team_home'] == df['team_favorite_id']

# Spread from HOME perspective (negative means home lays points)
def spread_home(row):
    s = row['spread_favorite']
    if pd.isna(s):
        return None
    return -s if row['home_is_favorite'] else s

df['spread_home'] = df.apply(spread_home, axis=1)

# Did home cover?
df['home_cover'] = (df['home_margin'] + df['spread_home']).astype(float) > 0

out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)
print('✅ Cleaned dataset saved to', out_path.resolve())
print(df.head(3).to_string(index=False))
