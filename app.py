import re
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_NAME = "DepositRadar"
APP_SUBTITLE = "ระบบทำนายโอกาสเปิดบัญชีเงินฝากประจำจากแคมเปญการตลาด"
MEMBERS = "น.ส.ขวัญฤทัย โพธิ์สุวรรณ, นายณรงค์ศักดิ์ ตันกุล, นายพีรพงษ์ สุวรรณคีรี"
DATASET_URL = "https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset"

BASE_DIR = Path(__file__).parent
MODEL_2F_PATH = BASE_DIR / "bank_tree_2f.joblib"
MODEL_8F_PATH = BASE_DIR / "bank_tree_8f.joblib"
CUSTOMER_PATH = BASE_DIR / "customers.csv"
HISTORY_PATH = BASE_DIR / "predictions.csv"

CUSTOMER_COLS = ["id", "ชื่อ-นามสกุล", "เพศ", "อายุ", "เบอร์โทร", "ที่อยู่", "บันทึกเมื่อ"]
HISTORY_COLS = [
    "id", "เวลา", "ผู้บันทึก", "ลูกค้า", "โมเดลที่ใช้",
    "ระยะเวลาโทร(วิ)", "วันตั้งแต่ติดต่อครั้งก่อน(pdays)",
    "อายุ", "ยอดเงินคงเหลือ", "จำนวนติดต่อแคมเปญนี้", "จำนวนติดต่อก่อนหน้า",
    "ผิดนัดชำระ", "สินเชื่อบ้าน", "สินเชื่อส่วนบุคคล",
    "ผลทำนาย", "ความน่าจะเป็น(%)",
]
MODEL_2F = "2 ฟีเจอร์ (duration, pdays)"
MODEL_8F = "8 ฟีเจอร์ (age, balance, duration, campaign, previous, default, housing, loan)"

USERS = {
    "admin": {"password": "1234", "display": "ผู้ดูแลระบบ"},
    "member": {"password": "1234", "display": "สมาชิกกลุ่ม"},
}

PAGE_DASH = "📊 แดชบอร์ด"
PAGE_PREDICT = "📡 ทำนายโอกาส"
PAGE_CUSTOMER = "🗂️ ข้อมูลลูกค้า"
PAGE_HISTORY = "🕘 ประวัติการทำนาย"
PAGE_ABOUT = "ℹ️ เกี่ยวกับระบบ"

st.set_page_config(page_title=f"{APP_NAME} | {APP_SUBTITLE}", page_icon="📡", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Prompt', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 1100px; }

.hero { background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #4338ca 100%);
    border-radius: 22px; padding: 30px 32px; color: #fff; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(15, 118, 110, .25); }
.hero .badge { display: inline-block; background: rgba(255,255,255,.18); padding: 4px 14px;
    border-radius: 999px; font-size: 13px; margin-bottom: 10px; letter-spacing: .5px; }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 700; color: #fff; padding: 0; }
.hero p  { margin: 6px 0 0 0; font-size: 1.02rem; opacity: .92; }

.section-title { font-weight: 600; font-size: 1.05rem; margin: 2px 0 10px 0; }

.result-card { border-radius: 20px; padding: 26px 28px; margin-top: 8px; color: #fff; }
.result-yes { background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 8px 24px rgba(16,185,129,.30); }
.result-no  { background: linear-gradient(135deg, #b45309, #f59e0b); box-shadow: 0 8px 24px rgba(245,158,11,.30); }
.result-card .label { font-size: .95rem; opacity: .9; }
.result-card .verdict { font-size: 1.6rem; font-weight: 700; margin: 2px 0 12px 0; }
.result-card .pct { font-size: 3rem; font-weight: 700; line-height: 1; }
.bar-bg { background: rgba(255,255,255,.28); border-radius: 999px; height: 12px; margin-top: 14px; overflow: hidden; }
.bar-fg { background: #fff; height: 100%; border-radius: 999px; }

.stat-box { border-radius: 16px; padding: 16px 18px; background: rgba(15,118,110,.08);
    border: 1px solid rgba(15,118,110,.25); height: 100%; }
.stat-box .n { font-size: 1.8rem; font-weight: 700; color: #0f766e; line-height: 1.1; }
.stat-box .t { font-size: .85rem; opacity: .8; }

.brand { text-align: center; padding: 10px 0 4px 0; }
.brand .logo { font-size: 2.4rem; }
.brand .name { font-size: 1.4rem; font-weight: 700; color: #0f766e; }
.brand .sub  { font-size: .78rem; opacity: .75; }

.footer { text-align: center; margin-top: 34px; padding: 18px 10px 6px 10px;
    border-top: 1px solid rgba(128,128,128,.25); font-size: .92rem; opacity: .85; }
.footer b { font-weight: 600; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    border-radius: 14px; height: 3rem; font-weight: 600; font-size: 1.02rem; }
.login-wrap { max-width: 380px; margin: 40px auto 0 auto; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 12px; border-radius: 12px; margin-bottom: 4px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(15,118,110,.10); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model(path_str: str):
    return joblib.load(Path(path_str))


# ช่วง min/max ที่ใช้ทำ Min-Max scaling ตอนเทรน (โมเดลเรียนจากข้อมูลสเกล 0-1)
# duration: ยืนยันแล้วจากข้อมูลจริงในสมุดโน้ต (ตรงกับค่าดั้งเดิมของชุดข้อมูลพอดี)
# pdays: ค่าต่ำสุด -1 หมายถึง "ยังไม่เคยถูกติดต่อมาก่อน" ค่าสูงสุดอ้างอิงจากชุดข้อมูล
#        Bank Marketing (11,162 แถว) มาตรฐาน — ควรตรวจสอบกับ df["pdays"].max() ก่อน scale จริงอีกครั้ง
# age / balance / campaign / previous: ช่วงของโมเดล 8 ฟีเจอร์ชุดเดิม
RANGES = {
    "duration": (2, 3881),
    "pdays": (-1, 854),
    "age": (18, 95),
    "balance": (-6847, 81204),
    "campaign": (1, 63),
    "previous": (0, 58),
}


def minmax(name: str, value: float) -> float:
    lo, hi = RANGES[name]
    return (value - lo) / (hi - lo)


def _load(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    return pd.DataFrame(columns=cols)


def _save(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_customers() -> pd.DataFrame:
    return _load(CUSTOMER_PATH, CUSTOMER_COLS)


def load_history() -> pd.DataFrame:
    return _load(HISTORY_PATH, HISTORY_COLS)


def add_customer(name, gender, age, phone, address) -> None:
    df = load_customers()
    new_id = int(df["id"].astype(int).max()) + 1 if len(df) else 1
    row = {
        "id": str(new_id), "ชื่อ-นามสกุล": name.strip(), "เพศ": gender, "อายุ": str(age),
        "เบอร์โทร": phone.strip(), "ที่อยู่": address.strip(),
        "บันทึกเมื่อ": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True), CUSTOMER_PATH)


def add_history(fields: dict) -> None:
    df = load_history()
    row = {"id": uuid.uuid4().hex[:8], "เวลา": datetime.now().strftime("%Y-%m-%d %H:%M"), **fields}
    _save(pd.concat([df, pd.DataFrame([row])], ignore_index=True), HISTORY_PATH)


def valid_phone(phone: str) -> bool:
    digits = re.sub(r"[\s-]", "", phone)
    return bool(re.fullmatch(r"0\d{8,9}", digits))


def login_screen() -> None:
    st.markdown(
        f'<div class="brand"><div class="logo">📡</div><div class="name">{APP_NAME}</div>'
        f'<div class="sub">{APP_SUBTITLE}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
    with st.form("login_form"):
        st.markdown("##### 🔐 เข้าสู่ระบบ")
        u = st.text_input("ชื่อผู้ใช้", placeholder="admin")
        p = st.text_input("รหัสผ่าน", type="password", placeholder="1234")
        ok = st.form_submit_button("เข้าสู่ระบบ", type="primary")
    st.caption("บัญชีสาธิต: admin / 1234 หรือ member / 1234")
    st.markdown("</div>", unsafe_allow_html=True)

    if ok:
        user = USERS.get(u)
        if user and user["password"] == p:
            st.session_state["auth_user"] = {"username": u, "display": user["display"]}
            st.rerun()
        else:
            st.error("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง")


def hero(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><div class="badge">{badge}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        f"""<div class="footer"><b>สมาชิก:</b> {MEMBERS}<br>
<span style="font-size:.82rem;">{APP_NAME} · โมเดลจากชุดข้อมูล Bank Marketing</span></div>""",
        unsafe_allow_html=True,
    )


def stat_box(col, n, label):
    with col:
        st.markdown(f'<div class="stat-box"><div class="n">{n}</div><div class="t">{label}</div></div>', unsafe_allow_html=True)


def page_dashboard() -> None:
    hero("📊 Overview", "แดชบอร์ดสรุปภาพรวม", "สรุปข้อมูลลูกค้าและผลการทำนายทั้งหมดในระบบ")

    customers, hist = load_customers(), load_history()
    yes = int((hist["ผลทำนาย"] == "มีแนวโน้มเปิดบัญชี").sum()) if len(hist) else 0
    rate = f"{yes / len(hist):.0%}" if len(hist) else "-"

    c1, c2, c3, c4 = st.columns(4)
    stat_box(c1, len(customers), "ลูกค้าในระบบ")
    stat_box(c2, len(hist), "การทำนายทั้งหมด")
    stat_box(c3, yes, "แนวโน้มเปิดบัญชี")
    stat_box(c4, rate, "อัตราแนวโน้มเปิดบัญชี")

    st.write("")
    if hist.empty:
        st.info("ยังไม่มีประวัติการทำนาย ลองไปที่หน้า **📡 ทำนายโอกาส** เพื่อเริ่มใช้งาน")
    else:
        left, right = st.columns([1.3, 1])
        with left:
            with st.container(border=True):
                st.markdown('<div class="section-title">📈 จำนวนการทำนายรายวัน</div>', unsafe_allow_html=True)
                daily = hist.copy()
                daily["วันที่"] = pd.to_datetime(daily["เวลา"]).dt.date
                st.bar_chart(daily.groupby("วันที่").size())
        with right:
            with st.container(border=True):
                st.markdown('<div class="section-title">🥧 สัดส่วนผลทำนาย</div>', unsafe_allow_html=True)
                st.bar_chart(hist["ผลทำนาย"].value_counts())

        with st.container(border=True):
            st.markdown('<div class="section-title">🕘 การทำนายล่าสุด</div>', unsafe_allow_html=True)
            st.dataframe(hist.sort_values("เวลา", ascending=False).head(5)[["เวลา", "ลูกค้า", "ผลทำนาย", "ความน่าจะเป็น(%)"]], hide_index=True)

    footer()


def page_predict() -> None:
    hero("📡 AI · Binary Classification · Decision Tree", "ทำนายโอกาสเปิดบัญชี", APP_SUBTITLE)

    model_choice = st.radio("เลือกโมเดลที่ใช้ทำนาย", [MODEL_2F, MODEL_8F], horizontal=True)
    model_path = MODEL_2F_PATH if model_choice == MODEL_2F else MODEL_8F_PATH

    try:
        model = load_model(str(model_path))
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์โมเดล: {model_path.name}\n\nให้วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ app.py")
        st.stop()

    customers = load_customers()
    NONE_OPT = "— ไม่ระบุลูกค้า —"
    options = [NONE_OPT] + [f"{r['id']} · {r['ชื่อ-นามสกุล']}" for _, r in customers.iterrows()]

    row = {}
    log_extra = {
        "อายุ": "", "ยอดเงินคงเหลือ": "", "จำนวนติดต่อแคมเปญนี้": "", "จำนวนติดต่อก่อนหน้า": "",
        "ผิดนัดชำระ": "", "สินเชื่อบ้าน": "", "สินเชื่อส่วนบุคคล": "",
        "ระยะเวลาโทร(วิ)": "", "วันตั้งแต่ติดต่อครั้งก่อน(pdays)": "",
    }

    with st.container(border=True):
        st.markdown('<div class="section-title">📝 ข้อมูลที่ใช้ทำนาย</div>', unsafe_allow_html=True)
        picked = st.selectbox("ลูกค้า (ไม่บังคับ — เลือกเพื่อให้ขึ้นชื่อในผลทำนาย/ประวัติ)", options)

        duration = st.slider("ระยะเวลาโทรคุยครั้งล่าสุด (วินาที)", 2, 3881, 300, step=10)
        st.caption(f"≈ {duration // 60} นาที {duration % 60} วินาที")
        row["duration"] = minmax("duration", duration)
        log_extra["ระยะเวลาโทร(วิ)"] = duration

        if model_choice == MODEL_2F:
            never_contacted = st.checkbox("ลูกค้ารายนี้ยังไม่เคยถูกติดต่อจากแคมเปญก่อนหน้าเลย", value=True)
            pdays = -1 if never_contacted else st.slider("จำนวนวันตั้งแต่ถูกติดต่อครั้งก่อนหน้า (pdays)", 0, 854, 30)
            if never_contacted:
                st.caption("pdays = -1 (ยังไม่เคยติดต่อ)")
            row["pdays"] = minmax("pdays", pdays)
            log_extra["วันตั้งแต่ติดต่อครั้งก่อน(pdays)"] = pdays
        else:
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("อายุ (ปี)", 18, 95, 40)
            with c2:
                balance = st.number_input("ยอดเงินคงเหลือในบัญชี (balance)", -6847, 81204, 1000, step=100)
            c3, c4, c5 = st.columns(3)
            with c3:
                default = st.radio("มีหนี้ผิดนัดชำระ", ["ไม่มี", "มี"], horizontal=True)
            with c4:
                housing = st.radio("มีสินเชื่อบ้าน", ["ไม่มี", "มี"], horizontal=True)
            with c5:
                loan = st.radio("มีสินเชื่อส่วนบุคคล", ["ไม่มี", "มี"], horizontal=True)
            campaign = st.slider("จำนวนครั้งที่ติดต่อในแคมเปญนี้", 1, 63, 2)
            previous = st.slider("จำนวนครั้งที่เคยติดต่อก่อนแคมเปญนี้", 0, 58, 0)

            row.update({
                "age": minmax("age", age), "balance": minmax("balance", balance),
                "campaign": minmax("campaign", campaign), "previous": minmax("previous", previous),
                "default": 1 if default == "มี" else 0, "housing": 1 if housing == "มี" else 0,
                "loan": 1 if loan == "มี" else 0,
            })
            log_extra.update({
                "อายุ": age, "ยอดเงินคงเหลือ": balance, "จำนวนติดต่อแคมเปญนี้": campaign,
                "จำนวนติดต่อก่อนหน้า": previous, "ผิดนัดชำระ": default,
                "สินเชื่อบ้าน": housing, "สินเชื่อส่วนบุคคล": loan,
            })

    if st.button("🔍 ทำนายโอกาส", type="primary"):
        X = pd.DataFrame([row])[list(model.feature_names_in_)]
        pred = int(model.predict(X)[0])
        p_yes = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

        if pred == 1:
            css, verdict, tip, verdict_plain = "result-yes", "✅ มีแนวโน้มเปิดบัญชีเงินฝากประจำ", "ควรให้ความสำคัญกับลูกค้ารายนี้ในการติดต่อต่อ", "มีแนวโน้มเปิดบัญชี"
        else:
            css, verdict, tip, verdict_plain = "result-no", "⚠️ มีโอกาสเปิดบัญชีค่อนข้างต่ำ", "อาจพิจารณาจัดลำดับความสำคัญของการติดต่อลงมา", "โอกาสต่ำ"

        cust_name = "ไม่ระบุ" if picked == NONE_OPT else picked.split(" · ", 1)[1]
        who = "ผลการทำนาย" if picked == NONE_OPT else f"ผลการทำนาย · ลูกค้า: {cust_name}"

        st.markdown(
            f"""<div class="result-card {css}"><div class="label">{who} · {model_choice.split(" (")[0]}</div>
<div class="verdict">{verdict}</div><div class="pct">{p_yes:.1%}</div>
<div class="label">โอกาสที่ลูกค้าจะเปิดบัญชี</div>
<div class="bar-bg"><div class="bar-fg" style="width:{p_yes * 100:.1f}%"></div></div>
<div class="label" style="margin-top:12px;">💡 {tip}</div></div>""",
            unsafe_allow_html=True,
        )

        add_history({
            "ผู้บันทึก": st.session_state["auth_user"]["display"],
            "ลูกค้า": cust_name,
            "โมเดลที่ใช้": model_choice.split(" (")[0],
            **log_extra,
            "ผลทำนาย": verdict_plain,
            "ความน่าจะเป็น(%)": f"{p_yes * 100:.1f}",
        })
        st.toast("บันทึกผลลงประวัติการทำนายแล้ว", icon="✅")

        with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล (หลังสเกลแล้ว)"):
            st.dataframe(X)

    footer()


def page_customer() -> None:
    hero("🗂️ Customer Records", "ข้อมูลลูกค้า", "บันทึกและจัดการรายชื่อลูกค้าของแคมเปญ")

    if msg := st.session_state.pop("flash", None):
        st.success(msg)

    with st.container(border=True):
        st.markdown('<div class="section-title">➕ เพิ่มลูกค้าใหม่</div>', unsafe_allow_html=True)
        with st.form("customer_form", clear_on_submit=True):
            name = st.text_input("ชื่อ-นามสกุล *", placeholder="เช่น สมชาย ใจดี")
            f1, f2, f3 = st.columns([1.2, 1, 1.6])
            with f1:
                gender = st.selectbox("เพศ", ["ชาย", "หญิง", "ไม่ระบุ"])
            with f2:
                age = st.number_input("อายุ (ปี)", 1, 120, 30)
            with f3:
                phone = st.text_input("เบอร์โทร *", placeholder="08xxxxxxxx", max_chars=12)
            address = st.text_area("ที่อยู่", placeholder="บ้านเลขที่ ถนน ตำบล อำเภอ จังหวัด รหัสไปรษณีย์")
            submitted = st.form_submit_button("💾 บันทึกข้อมูลลูกค้า", type="primary")

        if submitted:
            if not name.strip():
                st.error("กรุณากรอกชื่อ-นามสกุล")
            elif not valid_phone(phone):
                st.error("เบอร์โทรไม่ถูกต้อง ต้องขึ้นต้นด้วย 0 และมี 9-10 หลัก เช่น 0812345678")
            else:
                add_customer(name, gender, int(age), re.sub(r"[\s-]", "", phone), address)
                st.session_state["flash"] = f"บันทึกข้อมูลของ {name.strip()} เรียบร้อยแล้ว ✅"
                st.rerun()

    df = load_customers()
    s1, s2, s3 = st.columns(3)
    ages = pd.to_numeric(df["อายุ"], errors="coerce")
    stat_box(s1, len(df), "ลูกค้าทั้งหมด")
    stat_box(s2, f"{ages.mean():.0f}" if len(df) and ages.notna().any() else "-", "อายุเฉลี่ย (ปี)")
    fem = int((df["เพศ"] == "หญิง").sum())
    stat_box(s3, f"{fem} / {len(df) - fem}", "หญิง / อื่น ๆ")

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">📋 รายชื่อลูกค้าที่บันทึกไว้</div>', unsafe_allow_html=True)
        if df.empty:
            st.info("ยังไม่มีข้อมูลลูกค้า เริ่มเพิ่มรายแรกได้จากฟอร์มด้านบน")
        else:
            q = st.text_input("🔎 ค้นหา (ชื่อ / เบอร์โทร / ที่อยู่)", placeholder="พิมพ์เพื่อค้นหา")
            view = df
            if q.strip():
                mask = df[["ชื่อ-นามสกุล", "เบอร์โทร", "ที่อยู่"]].apply(
                    lambda col: col.str.contains(q.strip(), case=False, regex=False)
                ).any(axis=1)
                view = df[mask]
            st.dataframe(view, hide_index=True)
            st.caption(f"แสดง {len(view)} จาก {len(df)} รายการ")

            d1, d2 = st.columns(2)
            with d1:
                st.download_button("⬇️ ดาวน์โหลด CSV", df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                                    file_name="customers.csv", mime="text/csv")
            with d2:
                with st.popover("🗑️ ลบข้อมูลลูกค้า"):
                    labels = {f"{r['id']} · {r['ชื่อ-นามสกุล']}": r["id"] for _, r in df.iterrows()}
                    to_delete = st.multiselect("เลือกรายการที่ต้องการลบ", list(labels))
                    if st.button("ยืนยันการลบ", disabled=not to_delete):
                        ids = {labels[k] for k in to_delete}
                        _save(df[~df["id"].isin(ids)], CUSTOMER_PATH)
                        st.session_state["flash"] = f"ลบข้อมูล {len(ids)} รายการแล้ว"
                        st.rerun()

    footer()


def page_history() -> None:
    hero("🕘 Prediction Log", "ประวัติการทำนาย", "ทุกครั้งที่กดทำนาย ระบบจะบันทึกผลไว้ที่นี่โดยอัตโนมัติ")

    hist = load_history()
    if hist.empty:
        st.info("ยังไม่มีประวัติการทำนาย ลองไปที่หน้า **📡 ทำนายโอกาส**")
        footer()
        return

    with st.container(border=True):
        f1, f2 = st.columns(2)
        with f1:
            result_filter = st.selectbox("กรองตามผลทำนาย", ["ทั้งหมด", "มีแนวโน้มเปิดบัญชี", "โอกาสต่ำ"])
        with f2:
            q = st.text_input("ค้นหาชื่อลูกค้า", placeholder="พิมพ์เพื่อค้นหา")

        view = hist
        if result_filter != "ทั้งหมด":
            view = view[view["ผลทำนาย"] == result_filter]
        if q.strip():
            view = view[view["ลูกค้า"].str.contains(q.strip(), case=False, regex=False)]

        st.dataframe(view.sort_values("เวลา", ascending=False), hide_index=True)
        st.caption(f"แสดง {len(view)} จาก {len(hist)} รายการ")
        st.download_button("⬇️ ดาวน์โหลดประวัติทั้งหมด (CSV)",
                            hist.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                            file_name="predictions.csv", mime="text/csv")

    footer()


def page_about() -> None:
    hero("ℹ️ About", "เกี่ยวกับระบบ", "รายละเอียดโครงการและโมเดลที่ใช้")

    with st.container(border=True):
        st.markdown("#### 🎯 วัตถุประสงค์")
        st.write(
            "ระบบนี้ทำนายโอกาสที่ลูกค้าจะเปิดบัญชีเงินฝากประจำ (term deposit) "
            "จากแคมเปญการตลาดทางโทรศัพท์ของธนาคาร เพื่อช่วยให้ทีมการตลาดจัดลำดับ "
            "ความสำคัญในการติดต่อลูกค้าได้อย่างมีประสิทธิภาพมากขึ้น"
        )
    with st.container(border=True):
        st.markdown("#### 🧠 โมเดลที่ใช้ (เลือกได้ 2 แบบในหน้าทำนาย)")
        st.write("ทั้งสองแบบเป็น Decision Tree Classifier (max_depth=3) · Binary Classification: เปิดบัญชี / ไม่เปิดบัญชี")
        st.write(f"**{MODEL_2F}** — Accuracy บนชุดทดสอบ: 75.77%")
        st.write(f"**{MODEL_8F}** — Accuracy บนชุดทดสอบ: 75.64%")
        st.markdown(f"ชุดข้อมูลต้นฉบับ: [Bank Marketing Dataset]({DATASET_URL})")
    with st.container(border=True):
        st.markdown("#### 👥 สมาชิกกลุ่ม")
        for m in MEMBERS.split(", "):
            st.write(f"- {m}")

    footer()


def main():
    if "auth_user" not in st.session_state:
        login_screen()
        return

    with st.sidebar:
        st.markdown(
            f'<div class="brand"><div class="logo">📡</div><div class="name">{APP_NAME}</div>'
            f'<div class="sub">{APP_SUBTITLE}</div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(f"👋 สวัสดี, **{st.session_state['auth_user']['display']}**")
        page = st.radio("เมนู", [PAGE_DASH, PAGE_PREDICT, PAGE_CUSTOMER, PAGE_HISTORY, PAGE_ABOUT], label_visibility="collapsed")
        st.divider()
        st.caption(f"👥 ลูกค้า: **{len(load_customers())}** ราย · การทำนาย: **{len(load_history())}** ครั้ง")
        if st.button("🚪 ออกจากระบบ"):
            del st.session_state["auth_user"]
            st.rerun()

    {
        PAGE_DASH: page_dashboard,
        PAGE_PREDICT: page_predict,
        PAGE_CUSTOMER: page_customer,
        PAGE_HISTORY: page_history,
        PAGE_ABOUT: page_about,
    }[page]()


main()
