# 11_make_sqlite_data.py
# ---------------------------------------------------------------------------
# 1) 실제같은 데이터 생성
# 2) 같은 컬럼 스키마로 SQLite 파일에 저장
#
# 실행 방법:
#   python 11_make_sqlite_data.py
#
# 생성되는 파일:
#   company_news.db (테이블: news)
#
# 컬럼 스키마(현 데이터 컬럼과 동일):
#   - 기업명 (TEXT)
#   - 날짜 (TEXT, YYYY-MM-DD)
#   - 문서_카테고리 (TEXT)
#   - 요약 (TEXT)
#   - 주요_이벤트 (TEXT, JSON 문자열)
# ---------------------------------------------------------------------------

import sqlite3, json
from datetime import date

# 1. DB 경로/테이블명 정의 ----------------------------------------------------
DB_PATH = "company_news.db"
TABLE = "news"

# 2. 테이블 생성 SQL ----------------------------------------------------------
CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    기업명 TEXT NOT NULL,
    날짜 TEXT NOT NULL,
    문서_카테고리 TEXT NOT NULL,
    요약 TEXT NOT NULL,
    주요_이벤트 TEXT NOT NULL
);
"""


# 3. 샘플 데이터 생성(의미있는 내용) -----------------------------------------
def generate_records():
    return [
        {
            "기업명": "삼성전자",
            "날짜": "2025-03-15",
            "문서_카테고리": "사업전략",
            "요약": "삼성전자는 3나노 파운드리 수율 개선과 하반기 대형 고객사 수주 확대를 목표로 로드맵을 발표했다.",
            "주요_이벤트": ["수율개선", "수주확대"],
        },
        {
            "기업명": "삼성전자",
            "날짜": "2025-07-19",
            "문서_카테고리": "실적발표",
            "요약": "2025년 2분기 잠정 실적에서 반도체 부문 흑자 전환이 확인되었고, MX는 신제품 출시에 힘입어 견조한 수요를 기록했다.",
            "주요_이벤트": ["잠정실적", "흑자전환", "신제품"],
        },
        {
            "기업명": "현대차",
            "날짜": "2025-04-02",
            "문서_카테고리": "신제품",
            "요약": "현대차는 전기 SUV 신모델을 공개하며 10~80% 고속 충전 18분, 600km 이상 주행 성능을 강조했다. 북미 출시는 3분기 예정.",
            "주요_이벤트": ["신차공개", "해외출시"],
        },
        {
            "기업명": "네이버",
            "날짜": "2025-05-10",
            "문서_카테고리": "인수합병",
            "요약": "네이버는 해외 생성형 AI 스타트업의 일부 지분 인수를 검토 중이며, 검색·쇼핑 연계 서비스 고도화를 위해 기술 협력을 추진한다.",
            "주요_이벤트": ["지분인수", "AI협력"],
        },
        {
            "기업명": "카카오",
            "날짜": "2025-06-28",
            "문서_카테고리": "규제",
            "요약": "카카오는 금융 신서비스 출시를 앞두고 내부통제·개인정보 보호를 강화하는 정책을 발표했다.",
            "주요_이벤트": ["내부통제강화", "개인정보보호"],
        },
    ]


# 4. DB 초기화 및 데이터 저장 ------------------------------------------------
def init_and_seed(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(CREATE_SQL)
    conn.commit()

    # 테이블 비어있는 경우에만 삽입
    cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
    count = cur.fetchone()[0]
    if count == 0:
        for r in generate_records():
            cur.execute(
                f"""INSERT INTO {TABLE} (기업명, 날짜, 문서_카테고리, 요약, 주요_이벤트)
                      VALUES (?, ?, ?, ?, ?)""",
                (
                    r["기업명"],
                    r["날짜"],
                    r["문서_카테고리"],
                    r["요약"],
                    json.dumps(r["주요_이벤트"], ensure_ascii=False),
                ),
            )
        conn.commit()
        print(f"Seed 완료: {TABLE}에 {len(generate_records())}건 삽입")
    else:
        print(f"이미 데이터가 {count}건 존재합니다. 삽입 생략.")

    conn.close()


if __name__ == "__main__":
    init_and_seed(DB_PATH)
    print(f"SQLite 파일 생성/갱신 완료: {DB_PATH}")
