"""
סקריפט למילוי מחזורים בקבצי הפלייאוף הקיימים על פי הנתונים מ-BetExplorer
"""
import pandas as pd
from pathlib import Path
import re

def normalize_team_name(name):
    """נרמל שם קבוצה"""
    # הסר רווחים מיותרים
    name = str(name).strip()
    
    # תיקונים ידועים
    replacements = {
        "Hapoel Be'er Sheva": 'Hapoel Beer Sheva',
        'Ironi Kirya Shmona': 'Ironi Kiryat Shmona',
        'Kiryat Shmona': 'Ironi Kiryat Shmona',
        'Ramat Hasharon': 'Ironi Ramat HaSharon',
        'Ramat HaSharon': 'Ironi Ramat HaSharon',
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
    
    return name

def match_teams(home1, away1, home2, away2):
    """בדוק אם שתי קבוצות משחק תואמות"""
    h1 = normalize_team_name(home1)
    a1 = normalize_team_name(away1)
    h2 = normalize_team_name(home2)
    a2 = normalize_team_name(away2)
    
    return h1 == h2 and a1 == a2

def update_playoff_file(existing_file, scraped_file, playoff_type):
    """
    עדכן קובץ פלייאוף קיים עם מספרי מחזורים מהנתונים שנשלפו
    
    Args:
        existing_file: נתיב לקובץ הקיים
        scraped_file: נתיב לקובץ ששלפנו
        playoff_type: 'championship' או 'relegation'
    """
    if not existing_file.exists():
        print(f"   ⚠️ הקובץ {existing_file.name} לא קיים")
        return False
    
    if not scraped_file.exists():
        print(f"   ⚠️ הקובץ הנשלף {scraped_file.name} לא קיים")
        return False
    
    # קרא קבצים
    df_existing = pd.read_csv(existing_file)
    df_scraped = pd.read_csv(scraped_file)
    
    updates_count = 0
    
    # עבור על כל שורה בקובץ הקיים
    for idx, row in df_existing.iterrows():
        # דלג אם כבר יש מחזור
        if pd.notna(row.get('round')) and row.get('round') != '':
            continue
        
        home = row['home_team']
        away = row['away_team']
        home_goals = row['home_goals']
        away_goals = row['away_goals']
        
        # חפש התאמה בנתונים הנשלפים
        for _, scraped_row in df_scraped.iterrows():
            if (match_teams(home, away, scraped_row['home_team'], scraped_row['away_team']) and
                int(home_goals) == int(scraped_row['home_goals']) and
                int(away_goals) == int(scraped_row['away_goals'])):
                
                # מצאנו התאמה - עדכן מחזור
                df_existing.at[idx, 'round'] = int(scraped_row['round'])
                updates_count += 1
                break
    
    # שמור קובץ מעודכן
    if updates_count > 0:
        df_existing.to_csv(existing_file, index=False)
        print(f"   ✅ {existing_file.name}: עודכנו {updates_count} משחקים")
        return True
    else:
        print(f"   ℹ️ {existing_file.name}: לא נדרשו עדכונים")
        return False

def main():
    """עיבוד ראשי"""
    print("\n" + "="*70)
    print("🔄 מילוי מחזורים בקבצי פלייאוף קיימים")
    print("="*70 + "\n")
    
    # נתיבים
    scraped_dir = Path('data/playoffs/scraped_betexplorer')
    existing_dir = Path('data/playoffs')
    
    if not scraped_dir.exists():
        print(f"❌ תיקיית נתונים נשלפים לא קיימת: {scraped_dir}")
        return
    
    # מצא את כל הקבצים הנשלפים
    scraped_files = list(scraped_dir.glob('*.csv'))
    if not scraped_files:
        print(f"❌ לא נמצאו קבצים נשלפים ב-{scraped_dir}")
        return
    
    print(f"📁 נמצאו {len(scraped_files)} קבצים נשלפים")
    print(f"📁 תיקיית קבצים קיימים: {existing_dir}\n")
    
    # עבד כל קובץ נשלף
    total_updates = 0
    for scraped_file in sorted(scraped_files):
        # דלג על קבצים מאוחדים
        if 'all_' in scraped_file.name:
            continue
        
        # חלץ פרטי קובץ
        # championship_2012_13.csv -> season=2012_13, type=championship
        match = re.match(r'(championship|relegation)_(\d{4})_(\d{2})\.csv', scraped_file.name)
        if not match:
            continue
        
        playoff_type = match.group(1)
        season_year = match.group(2)
        
        # מצא קובץ קיים מתאים
        existing_pattern = f"playoffs_{playoff_type}_{season_year}_{match.group(3)}_ligat_haal_wikipedia.csv"
        existing_file = existing_dir / existing_pattern
        
        print(f"\n📄 מעבד: {scraped_file.name}")
        print(f"   🎯 קובץ יעד: {existing_pattern}")
        
        if update_playoff_file(existing_file, scraped_file, playoff_type):
            total_updates += 1
    
    print("\n" + "="*70)
    print(f"✨ הסתיים! עודכנו {total_updates} קבצים")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
