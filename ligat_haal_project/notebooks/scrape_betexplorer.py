"""
סקריפט לחילוץ נתוני משחקי פלייאוף מאתר BetExplorer
"""
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def scrape_betexplorer_playoffs(url, season="2012/13"):
    """
    חילוץ נתוני משחקי פלייאוף מ-BetExplorer
    
    Args:
        url: כתובת ה-URL של העונה
        season: שם העונה
    """
    # הגדרות Chrome
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # ריצה ברקע - כבוי לבדיקה
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    print(f"מתחיל סקרייפינג של {url}")
    
    try:
        # יצירת driver
        driver = webdriver.Chrome(options=chrome_options)
        
        # גישה לדף התוצאות
        results_url = url + "results/?stage=fkMHNw24"  # Championship Group
        print(f"גושה ל-{results_url}")
        driver.get(results_url)
        
        # המתן לטעינת העמוד
        time.sleep(5)
        
        # חילוץ HTML
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # נסיון למצוא את כל המחזורים (rounds)
        matches_data = []
        
        # חיפוש כל קישורי המשחקים
        match_links = soup.find_all('a', href=lambda x: x and '/football/' in str(x) and '-' in str(x) and x.count('/') >= 7)
        print(f"נמצאו {len(match_links)} קישורי משחקים")
        
        # ניסיון למצוא מבנה של מחזורים
        # BetExplorer משתמש במבנה עם headers למחזורים
        headers = soup.find_all(['h3', 'h4', 'div'], class_=lambda x: x and ('round' in str(x).lower() or 'header' in str(x).lower()))
        print(f"נמצאו {len(headers)} כותרות אפשריות")
        
        # שמירת HTML מעודכן לבדיקה
        with open('betexplorer_results_page.html', 'w', encoding='utf-8') as f:
            f.write(page_source)
        print("HTML של דף התוצאות נשמר לקובץ betexplorer_results_page.html לבדיקה")
        
        # נסיון למצוא טבלאות עם תוצאות
        tables = soup.find_all('table')
        print(f"נמצאו {len(tables)} טבלאות בדף התוצאות")
        
        for idx, table in enumerate(tables[:3]):  # בדוק את 3 הטבלאות הראשונות
            print(f"\nטבלה {idx+1}:")
            rows = table.find_all('tr')
            print(f"  - {len(rows)} שורות")
            if rows:
                first_row = rows[0]
                print(f"  - תוכן שורה ראשונה: {first_row.text[:100]}")
        
        driver.quit()
        
        return matches_data
        
    except Exception as e:
        print(f"שגיאה: {e}")
        if 'driver' in locals():
            driver.quit()
        return []

def main():
    url = "https://www.betexplorer.com/football/israel/ligat-ha-al-2012-2013/"
    
    print("מתחיל חילוץ נתונים...")
    print("=" * 60)
    
    matches = scrape_betexplorer_playoffs(url)
    
    if matches:
        df = pd.DataFrame(matches)
        print(f"\nנאספו {len(matches)} משחקים")
        print(df.head())
        
        # שמירה לקובץ
        output_file = "data/playoffs/betexplorer_playoffs_2012_13.csv"
        df.to_csv(output_file, index=False)
        print(f"\nהנתונים נשמרו ל-{output_file}")
    else:
        print("\nלא נמצאו נתונים. בדוק את הקובץ betexplorer_page.html לניתוח המבנה")

if __name__ == "__main__":
    main()
