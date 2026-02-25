import requests
from bs4 import BeautifulSoup
import time
import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError


BASE_URL = "https://www.10000recipe.com"
LIST_URL = "https://www.10000recipe.com/recipe/list.html"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

REQUEST_DELAY = 0.7
MAX_PAGES = 100

MONGO_URI = "mongodb://root:RootPassword123@mongodb:27017/admin"
DB_NAME = "recipe_db"
RECIPE_COL = "recipes"
META_COL = "crawl_meta"

# 로그 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
session = requests.Session()
session.headers.update(HEADERS)


# MongoDB
def get_mongo():
    mongo = MongoClient(MONGO_URI)
    db = mongo[DB_NAME]
    recipes = db[RECIPE_COL]
    meta = db[META_COL]

    recipes.create_index([("recipe_id", ASCENDING)], unique=True)

    return recipes, meta


# 메타 정보
def get_last_crawled_at(meta):
    doc = meta.find_one({"source": "10000recipe"})
    if not doc:
        return None
    return datetime.fromisoformat(doc["last_crawled_at"])


def update_last_crawled_at(meta, dt: datetime):
    meta.update_one(
        {"source": "10000recipe"},
        {"$set": {"last_crawled_at": dt.isoformat()}},
        upsert=True,
    )


# 리스트 페이지 → recipe_id 수집
def get_recipe_ids_by_page(page: int):
    res = session.get(LIST_URL, params={"order": "date", "page": page}, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    recipe_ids = []
    for a in soup.select("a.common_sp_link"):
        href = a.get("href", "")
        if href.startswith("/recipe/"):
            recipe_ids.append(href.split("/")[-1])

    return list(dict.fromkeys(recipe_ids))


# 상세 페이지 크롤링
def crawl_recipe_detail(recipe_id: str):
    url = f"{BASE_URL}/recipe/{recipe_id}"
    res = session.get(url, timeout=10)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    author = soup.select_one("span.user_info2_name")
    title = soup.select_one("div.view2_summary.st3 h3")
    intro = soup.select_one("#recipeIntro")
    img = soup.select_one("#main_thumbs")

    portion = soup.select_one(".view2_summary_info1")
    cook_time = soup.select_one(".view2_summary_info2")
    level = soup.select_one(".view2_summary_info3")

    # 등록일 / 수정일
    registered_at = modified_at = None
    date_p = soup.select_one("p.view_notice_date")
    if date_p:
        dates = date_p.select("b")
        if len(dates) > 0:
            registered_at = (
                dates[0].get_text(strip=True).replace("등록일 :", "").strip()
            )
        if len(dates) > 1:
            modified_at = dates[1].get_text(strip=True).replace("수정일 :", "").strip()

    # 재료
    ingredients = []
    box = soup.select_one("div.ready_ingre3#divConfirmedMaterialArea")
    if box:
        for li in box.select("li"):
            name_tag = li.select_one(".ingre_list_name a")
            desc_tag = li.select_one(".ingre_list_name span")
            amount_tag = li.select_one(".ingre_list_ea")

            if not name_tag:
                continue

            ingredients.append(
                {
                    "name": name_tag.get_text(" ", strip=True),
                    "desc": desc_tag.get_text(strip=True) if desc_tag else None,
                    "amount": amount_tag.get_text(strip=True) if amount_tag else None,
                }
            )
    # 케이스 2: div.cont_ingre
    else:
        cont_ingre = soup.select_one("div.cont_ingre")
        if cont_ingre:
            for dl in cont_ingre.select("dl"):
                dt = dl.select_one("dt")
                dd = dl.select_one("dd")

                if not dd:
                    continue

                category = dt.get_text(strip=True) if dt else "재료"
                text = dd.get_text(" ", strip=True)

                items = [item.strip() for item in text.split(",") if item.strip()]

                for item in items:
                    # 마지막 2개 단어만 체크
                    words = item.split()

                    # 마지막 단어에 숫자가 있으면 수량으로 간주
                    if len(words) >= 2 and any(char.isdigit() for char in words[-1]):
                        # "마늘 2톨" 또는 "요구르트 300 g" 처리
                        if len(words) >= 3 and any(
                            char.isdigit() for char in words[-2]
                        ):
                            # "크림치즈 200 g" → name="크림치즈", amount="200 g"
                            name = " ".join(words[:-2])
                            amount = " ".join(words[-2:])
                        else:
                            # "마늘 2톨" → name="마늘", amount="2톨"
                            name = " ".join(words[:-1])
                            amount = words[-1]

                        ingredients.append(
                            {
                                "name": name,
                                "desc": None,
                                "amount": amount,
                                "category": category,
                            }
                        )
                    else:
                        # 숫자 없으면 전체를 name으로
                        ingredients.append(
                            {
                                "name": item,
                                "desc": None,
                                "amount": None,
                                "category": category,
                            }
                        )

    # 조리 단계
    steps = []
    for idx, div in enumerate(soup.select("div[id^=stepdescr]"), start=1):
        steps.append(f"{idx}. {div.get_text(strip=True)}")

    registered_at_dt = (
        datetime.strptime(registered_at, "%Y-%m-%d") if registered_at else None
    )

    modified_at_dt = datetime.strptime(modified_at, "%Y-%m-%d") if modified_at else None

    return {
        "recipe_id": recipe_id,
        "detail_url": url,
        "author": author.get_text(strip=True) if author else None,
        "title": title.get_text(strip=True) if title else None,
        "intro": intro.get_text(" ", strip=True) if intro else None,
        "image": img["src"] if img else None,
        "portion": portion.get_text(strip=True) if portion else None,
        "cook_time": cook_time.get_text(strip=True) if cook_time else None,
        "level": level.get_text(strip=True) if level else None,
        "registered_at": registered_at_dt,
        "modified_at": modified_at_dt,
        "ingredients": ingredients,
        "steps": steps,
    }


# 페이지 단위 크롤링 + 시간 단위로
def crawl_incremental():
    recipes, meta = get_mongo()
    last_crawled_at = get_last_crawled_at(meta)
    logger.info(f"LAST_CRAWLED_AT = {last_crawled_at}")

    newest_seen = last_crawled_at

    for page in range(1, MAX_PAGES + 1):
        recipe_ids = get_recipe_ids_by_page(page)
        logger.info(f"PAGE {page} | {len(recipe_ids)} recipes")

        should_stop = False

        for rid in recipe_ids:
            try:
                data = crawl_recipe_detail(rid)

                # 데이터가 없으면 스킵
                if not data:
                    logger.info(f"SKIP | no data | {rid}")
                    continue

                # newest_seen 업데이트 (재료 유무와 무관하게!)
                if data.get("registered_at"):
                    if not newest_seen or data["registered_at"] > newest_seen:
                        newest_seen = data["registered_at"]
                        logger.info(f"UPDATE newest_seen | {newest_seen} | {rid}")

                # 이전 크롤링보다 오래된 데이터면 중단
                if (
                    last_crawled_at
                    and data.get("registered_at")
                    and data["registered_at"] < last_crawled_at
                ):
                    logger.info(f"STOP | reached old data | {data['registered_at']}")
                    should_stop = True
                    break

                # 재료가 없으면 저장하지 않음
                if not data.get("ingredients"):
                    logger.info(f"SKIP | no ingredients | {rid}")
                    time.sleep(REQUEST_DELAY)
                    continue

                try:
                    recipes.insert_one(data)
                    logger.info(f"SAVED | recipe_id={rid} | {data.get('title', 'N/A')}")

                except DuplicateKeyError:
                    logger.info(f"SKIP | already exists | {rid}")

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.error(f"FAIL | {rid} | {e}")
                time.sleep(REQUEST_DELAY)

        if should_stop:
            break

    # 메타 업데이트
    if newest_seen:
        if newest_seen != last_crawled_at:
            update_last_crawled_at(meta, newest_seen)
            logger.info(f"UPDATED META | newest_seen = {newest_seen}")
        else:
            logger.info(f"NO CHANGE | newest_seen = {newest_seen}")
    else:
        logger.info(f"NO newest_seen to update")
