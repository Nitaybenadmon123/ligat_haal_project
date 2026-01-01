"""
סקריפט מקיף לחילוץ נתוני פלייאוף מ-BetExplorer לכל העונות
"""
import time
import pandas as pd
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from pathlib import Path

# מיפוי שמות קבוצות
TEAM_NAME_MAP = {
    'Maccabi Tel Aviv': 'Maccabi Tel Aviv',
    'Maccabi Haifa': 'Maccabi Haifa',
    'Hapoel Tel Aviv': 'Hapoel Tel Aviv',
    'Hapoel Be\'er Sheva': 'Hapoel Beer Sheva',
    'Hapoel Beer Sheva': 'Hapoel Beer Sheva',
    'Beitar Jerusalem': 'Beitar Jerusalem',
    'Bnei Yehuda': 'Bnei Yehuda',
    'Maccabi Netanya': 'Maccabi Netanya',
    'Kiryat Shmona': 'Ironi Kiryat Shmona',
    'Ramat Hasharon': 'Ironi Ramat HaSharon',
    'Ironi Kiryat Shmona': 'Ironi Kiryat Shmona',
    'Hapoel Haifa': 'Hapoel Haifa',
    'Maccabi Petah Tikva': 'Maccabi Petah Tikva',
    'Hapoel Raanana': 'Hapoel Raanana',
    'Bnei Sakhnin': 'Bnei Sakhnin',
    'Ashdod': 'Ashdod SC',
    'Hapoel Ashkelon': 'Hapoel Ashkelon',
    'Hapoel Acre': 'Hapoel Acre',
    'Hapoel Kfar Saba': 'Hapoel Kfar Saba',
    'Maccabi Ahi Nazareth': 'Maccabi Ahi Nazareth',
}

def normalize_team_name(name):
    """נרמל שם קבוצה"""
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)

def find_playoff_stages(driver, base_url):
    """
    מוצא את ה-stage IDs של הפלייאופים בעמוד
    
    Returns:
        dict: {'championship': stage_id, 'relegation': stage_id}
    """
    stages = {}
    
    try:
        # גש לדף הראשי של העונה
        driver.get(base_url)
        time.sleep(2)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # חפש את הלינקים לפלייאופים בתפריט
        # בדף העונה יש לחצנים/לינקים לקבוצות השונות
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link.get('href', '')
            text = link.text.strip().lower()
            
            # זיהוי פלייאוף עליון
            if 'championship' in text or 'top' in text or 'עליון' in text:
                if 'stage=' in href:
                    stage_match = re.search(r'stage=([^&]+)', href)
                    if stage_match:
                        stages['championship'] = stage_match.group(1)
                        print(f"   ✓ נמצא פלייאוף עליון: stage={stages['championship']}")
            
            # זיהוי פלייאוף תחתון
            elif 'relegation' in text or 'bottom' in text or 'תחתון' in text:
                if 'stage=' in href:
                    stage_match = re.search(r'stage=([^&]+)', href)
                    if stage_match:
                        stages['relegation'] = stage_match.group(1)
                        print(f"   ✓ נמצא פלייאוף תחתון: stage={stages['relegation']}")
        
        # אם לא נמצא דרך הלינקים, נסה למצוא בדרך אחרת
        # לפעמים זה בתפריט dropdown
        if not stages:
            select_elements = soup.find_all('select')
            for select in select_elements:
                options = select.find_all('option')
                for option in options:
                    value = option.get('value', '')
                    text = option.text.strip().lower()
                    
                    if 'championship' in text or 'top' in text:
                        if 'stage=' in value:
                            stage_match = re.search(r'stage=([^&]+)', value)
                            if stage_match:
                                stages['championship'] = stage_match.group(1)
                    
                    elif 'relegation' in text or 'bottom' in text:
                        if 'stage=' in value:
                            stage_match = re.search(r'stage=([^&]+)', value)
                            if stage_match:
                                stages['relegation'] = stage_match.group(1)
    
    except Exception as e:
        print(f"   ⚠️ שגיאה בחיפוש stage IDs: {e}")
    
    return stages

def scrape_season_playoffs(season_year, season_format="2012-2013"):
    """
    חילוץ נתוני פלייאוף לעונה ספציפית
    
    Args:
        season_year: שנה (2012)
        season_format: פורמט בURL (2012-2013)
    
    Returns:
        dict with 'championship' and 'relegation' DataFrames
    """
    base_url = f"https://www.betexplorer.com/football/israel/ligat-ha-al-{season_format}/"
    results_url = base_url + "results/"
    
    print(f"\n{'='*70}")
    print(f"מחלץ עונה {season_format}")
    print(f"{'='*70}")
    
    # הגדרות Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    
    driver = None
    results = {'championship': [], 'relegation': []}
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        
        # מצא את ה-stage IDs של הפלייאופים
        print("\n🔍 מחפש stage IDs...")
        stages = find_playoff_stages(driver, base_url)
        
        if not stages.get('championship') and not stages.get('relegation'):
            print("   ⚠️ לא נמצאו stage IDs, ננסה עם ברירת מחדל...")
            stages = {'championship': 'fkMHNw24', 'relegation': 'ndqjLGnm'}
        
        # חילוץ פלייאוף עליון (Championship)
        if stages.get('championship'):
            print(f"\n📊 מחלץ פלייאוף עליון (stage={stages['championship']})...")
            champ_url = results_url + f"?stage={stages['championship']}"
            driver.get(champ_url)
            time.sleep(3)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            championship_matches = extract_matches_from_page(soup, season_year, 'championship')
            results['championship'] = championship_matches
            print(f"   נמצאו {len(championship_matches)} משחקים בפלייאוף עליון")
        
        # חילוץ פלייאוף תחתון (Relegation)
        if stages.get('relegation'):
            print(f"\n📊 מחלץ פלייאוף תחתון (stage={stages['relegation']})...")
            rel_url = results_url + f"?stage={stages['relegation']}"
            driver.get(rel_url)
            time.sleep(3)
            
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            relegation_matches = extract_matches_from_page(soup, season_year, 'relegation')
            results['relegation'] = relegation_matches
            print(f"   נמצאו {len(relegation_matches)} משחקים בפלייאוף תחתון")
        
    except Exception as e:
        print(f"❌ שגיאה בעונה {season_format}: {e}")
    
    finally:
        if driver:
            driver.quit()
    
    return results

def extract_matches_from_page(soup, season_year, playoff_type):
    """חילוץ משחקים מדף אחד"""
    matches = []
    current_round = None
    
    # מצא את הטבלה הראשית
    table = soup.find('table', class_='table-main')
    if not table:
        print("   ⚠️ לא נמצאה טבלה")
        return matches
    
    rows = table.find_all('tr')
    
    for row in rows:
        # בדוק אם זה כותרת מחזור
        th = row.find('th', class_='h-text-left')
        if th and 'Round' in th.text:
            # חלץ מספר מחזור
            round_text = th.text.strip()
            round_match = re.search(r'(\d+)\.\s*Round', round_text)
            if round_match:
                current_round = int(round_match.group(1))
                print(f"   🔄 מחזור {current_round}")
            continue
        
        # בדוק אם זה משחק
        match_link = row.find('a', class_='in-match')
        if not match_link:
            continue
        
        # חלץ שמות קבוצות
        spans = match_link.find_all('span')
        if len(spans) < 2:
            continue
        
        home_team = normalize_team_name(spans[0].text.strip())
        away_team = normalize_team_name(spans[1].text.strip())
        
        # חלץ תוצאה
        score_td = row.find('td', class_='h-text-center')
        if not score_td or not score_td.find('a'):
            continue
        
        score_text = score_td.find('a').text.strip()
        score_match = re.match(r'(\d+):(\d+)', score_text)
        if not score_match:
            continue
        
        home_goals = int(score_match.group(1))
        away_goals = int(score_match.group(2))
        goal_diff = home_goals - away_goals
        
        # קבע תוצאה
        if home_goals > away_goals:
            result = 'H'
            home_points = 3
            away_points = 0
        elif home_goals < away_goals:
            result = 'A'
            home_points = 0
            away_points = 3
        else:
            result = 'D'
            home_points = 1
            away_points = 1
        
        # בנה רשומה
        match = {
            'season': f"{season_year}/{str(int(season_year) + 1)[-2:]}",
            'season_year': int(season_year),
            'round': current_round,
            'playoff_type': playoff_type,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'goal_diff': goal_diff,
            'result': result,
            'home_points': home_points,
            'away_points': away_points
        }
        
        matches.append(match)
    
    return matches

def main():
    """עיבוד ראשי"""
    # רשימת עונות לעיבוד
    seasons = [
        (2009, '2009-2010'),
        (2010, '2010-2011'),
        (2011, '2011-2012'),
        (2012, '2012-2013'),
        (2013, '2013-2014'),
        (2014, '2014-2015'),
        (2015, '2015-2016'),
        (2016, '2016-2017'),
        (2017, '2017-2018'),
        (2018, '2018-2019'),
        (2019, '2019-2020'),
        (2020, '2020-2021'),
        (2021, '2021-2022'),
        (2022, '2022-2023'),
        (2023, '2023-2024'),
        (2024, '2024-2025'),
    ]
    
    output_dir = Path('data/playoffs/scraped_betexplorer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 מתחיל חילוץ נתוני פלייאוף מ-BetExplorer")
    print(f"📁 תיקיית פלט: {output_dir}")
    print(f"📅 {len(seasons)} עונות לעיבוד")
    
    all_championship = []
    all_relegation = []
    
    for season_year, season_format in seasons:
        try:
            results = scrape_season_playoffs(season_year, season_format)
            
            # שמור עונה בודדת
            if results['championship']:
                df_champ = pd.DataFrame(results['championship'])
                filename = output_dir / f'championship_{season_year}_{int(str(season_year)[-2:])+1}.csv'
                df_champ.to_csv(filename, index=False)
                print(f"   ✅ נשמר: {filename.name}")
                all_championship.extend(results['championship'])
            
            if results['relegation']:
                df_rel = pd.DataFrame(results['relegation'])
                filename = output_dir / f'relegation_{season_year}_{int(str(season_year)[-2:])+1}.csv'
                df_rel.to_csv(filename, index=False)
                print(f"   ✅ נשמר: {filename.name}")
                all_relegation.extend(results['relegation'])
            
            # המתנה בין עונות
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ שגיאה בעונה {season_format}: {e}")
            continue
    
    # שמור קבצים מאוחדים
    if all_championship:
        df_all_champ = pd.DataFrame(all_championship)
        all_champ_file = output_dir / 'all_championship_betexplorer.csv'
        df_all_champ.to_csv(all_champ_file, index=False)
        print(f"\n✅ נשמר קובץ מאוחד: {all_champ_file.name} ({len(all_championship)} משחקים)")
    
    if all_relegation:
        df_all_rel = pd.DataFrame(all_relegation)
        all_rel_file = output_dir / 'all_relegation_betexplorer.csv'
        df_all_rel.to_csv(all_rel_file, index=False)
        print(f"✅ נשמר קובץ מאוחד: {all_rel_file.name} ({len(all_relegation)} משחקים)")
    
    print(f"\n{'='*70}")
    print(f"✨ הסתיים בהצלחה!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
