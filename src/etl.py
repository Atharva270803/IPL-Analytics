import os
import yaml
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

os.chdir(r'C:\Users\Lenovo\ipl-analytics')
load_dotenv()

def get_engine():
    url = os.getenv("DATABASE_URL")
    return create_engine(url)

def parse_match(filepath):
    with open(filepath, encoding='utf-8') as f:
        data = yaml.safe_load(f)

    info = data['info']
    match_id = os.path.basename(filepath).replace('.yaml', '')
    winner = info.get('outcome', {}).get('winner', 'No result')
    toss = info.get('toss', {})

    deliveries = []
    inning_names = ['1st innings', '2nd innings']

    for inning_idx, inning_name in enumerate(inning_names):
        # Find this inning in the innings list
        inning_data = None
        for inn in data.get('innings', []):
            if inning_name in inn:
                inning_data = inn[inning_name]
                break
        if inning_data is None:
            continue

        batting_team = inning_data.get('team', '')

        for delivery_dict in inning_data.get('deliveries', []):
            # Each delivery_dict is like {0.1: {...}}
            for over_ball, details in delivery_dict.items():
                over_ball_str = str(over_ball)
                try:
                    over = int(float(over_ball_str))
                    ball = round((float(over_ball_str) - over) * 10)
                except:
                    over, ball = 0, 0

                runs = details.get('runs', {})
                wicket = details.get('wicket', None)

                deliveries.append({
                    'match_id':      match_id,
                    'inning':        inning_idx + 1,
                    'over':          over,
                    'ball':          ball,
                    'batting_team':  batting_team,
                    'batter':        details.get('batsman', ''),
                    'bowler':        details.get('bowler', ''),
                    'runs_batter':   runs.get('batsman', 0),
                    'runs_extras':   runs.get('extras', 0),
                    'runs_total':    runs.get('total', 0),
                    'is_wicket':     1 if wicket else 0,
                    'venue':         info.get('venue', ''),
                    'city':          info.get('city', ''),
                    'winner':        winner,
                    'toss_winner':   toss.get('winner', ''),
                    'toss_decision': toss.get('decision', ''),
                    'teams':         str(info.get('teams', [])),
                })
    return deliveries

def load_all(data_dir='data/raw'):
    engine = get_engine()
    all_rows = []
    files = [f for f in os.listdir(data_dir) if f.endswith('.yaml')]
    total = len(files)

    print(f"Parsing {total} match files...")
    skipped = 0
    for i, fname in enumerate(files):
        try:
            rows = parse_match(os.path.join(data_dir, fname))
            all_rows.extend(rows)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"Skipped {fname}: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Parsed {i+1}/{total} files — {len(all_rows):,} deliveries so far...")

    print(f"\nTotal deliveries parsed: {len(all_rows):,}")
    print(f"Files skipped: {skipped}")

    df = pd.DataFrame(all_rows)
    print(f"Unique venues: {df['venue'].nunique()}")
    print(f"Unique teams: {df['batting_team'].nunique()}")

    print("\nLoading into PostgreSQL...")
    df.to_sql('deliveries', engine, if_exists='replace',
              index=False, chunksize=1000)
    print("Done. Deliveries table created in ipl_db.")
    return df

if __name__ == '__main__':
    df = load_all()
    print(df.head())
    print(df.shape)
