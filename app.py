import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_NAME = "DepositRadar"
APP_SUBTITLE = "ระบบทำนายโอกาสเปิดบัญชีเงินฝากประจำ"
MEMBERS = "น.ส.ขวัญฤทัย โพธิ์สุวรรณ, นายณรงค์ศักดิ์ ตันกุล, นายพีรพงษ์ สุวรรณคีรี"

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "bank_tree.joblib"
CUSTOMER_PATH = BASE_DIR / "customers.csv"  # ไฟล์เก็บข้อมูลลูกค้า (สร้างอัตโนมัติ)
CUSTOMER_COLS = ["id", "ชื่อ-นามสกุล", "เพศ", "อายุ", "เบอร์โทร", "ที่อยู่", "บันทึกเมื่อ"]

PAGE_PREDICT = "📡  ทำนายโอกาส"
PAGE_CUSTOMER = "🗂️  ข้อมูลลูกค้า"

st.set_page_config(page_title=f"{APP_NAME} | {APP_SUBTITLE}", page_icon="📡", layout="centered")

# ---------------------------------------------------------------
# สไตล์ (CSS)
# ---------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Prompt', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; max-width: 860px; }

.hero {
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #4338ca 100%);
    border-radius: 22px; padding: 30px 32px; color: #fff; margin-bottom: 22px;
    box-shadow: 0 10px 30px rgba(15, 118, 110, .25);
}
.hero .badge {
    display: inline-block; background: rgba(255,255,255,.18); padding: 4px 14px;
    border-radius: 999px; font-size: 13px; margin-bottom: 10px; letter-spacing: .5px;
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 700; color: #fff; padding: 0; }
.hero p  { margin: 6px 0 0 0; font-size: 1.05rem; opacity: .92; }

.section-title { font-weight: 600; font-size: 1.05rem; margin: 2px 0 10px 0; }

.result-card { border-radius: 20px; padding: 26px 28px; margin-top: 8px; color: #fff; }
.result-yes { background: linear-gradient(135deg, #059669, #10b981); box-shadow: 0 8px 24px rgba(16,185,129,.30); }
.result-no  { background: linear-gradient(135deg, #b45309, #f59e0b); box-shadow: 0 8px 24px rgba(245,158,11,.30); }
.result-card .label { font-size: .95rem; opacity: .9; }
.result-card .verdict { font-size: 1.6rem; font-weight: 700; margin: 2px 0 12px 0; }
.result-card .pct { font-size: 3rem; font-weight: 700; line-height: 1; }
.bar-bg { background: rgba(255,255,255,.28); border-radius: 999px; height: 12px; margin-top: 14px; overflow: hidden; }
.bar-fg { background: #fff; height: 100%; border-radius: 999px; }

.stat-box {
    border-radius: 16px; padding: 16px 18px; background: rgba(15,118,110,.08);
    border: 1px solid rgba(15,118,110,.25);
}
.stat-box .n { font-size: 1.8rem; font-weight: 700; color: #0f766e; line-height: 1.1; }
.stat-box .t { font-size: .88rem; opacity: .8; }

.brand { text-align: center; padding: 6px 0 4px 0; }
.brand .logo { font-size: 2.4rem; }
.brand .name { font-size: 1.4rem; font-weight: 700; color: #0f766e; }
.brand .sub  { font-size: .8rem; opacity: .75; }

.footer {
    text-align: center; margin-top: 34px; padding: 18px 10px 6px 10px;
    border-top: 1px solid rgba(128,128,128,.25); font-size: .92rem; opacity: .85;
}
.footer b { font-weight: 600; }
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%; border-radius: 14px; height: 3.1rem; font-weight: 600; font-size: 1.05rem;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 10px 12px; border-radius: 12px; margin-bottom: 4px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background: rgba(15,118,110,.10); }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# โมเดล
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ช่วง min/max ที่ใช้ทำ Min-Max scaling ตอนเทรน (โมเดลเรียนจากข้อมูลสเกล 0-1)
# ถ้าใน Colab สเกลด้วยช่วงอื่น ให้แก้ตัวเลขตรงนี้ให้ตรงกับที่ใช้เทรน
RANGES = {
    "age": (18, 95),
    "balance": (-6847, 81204),
    "duration": (2, 3881),
    "campaign": (1, 63),
    "previous": (0, 58),
}


def minmax(name: str, value: float) -> float:
    lo, hi = RANGES[name]
    return (value - lo) / (hi - lo)


# ---------------------------------------------------------------
# ข้อมูลลูกค้า (เก็บเป็นไฟล์ CSV)
# ---------------------------------------------------------------
def load_customers() -> pd.DataFrame:
    if CUSTOMER_PATH.exists():
        return pd.read_csv(CUSTOMER_PATH, dtype=str, encoding="utf-8-sig").fillna("")
    return pd.DataFrame(columns=CUSTOMER_COLS)


def save_customers(df: pd.DataFrame) -> None:
    # utf-8-sig ทำให้เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน
    df.to_csv(CUSTOMER_PATH, index=False, encoding="utf-8-sig")


def add_customer(name, gender, age, phone, address) -> None:
    df = load_customers()
    new_id = int(df["id"].astype(int).max()) + 1 if len(df) else 1
    row = {
        "id": str(new_id),
        "ชื่อ-นามสกุล": name.strip(),
        "เพศ": gender,
        "อายุ": str(age),
        "เบอร์โทร": phone.strip(),
        "ที่อยู่": address.strip(),
        "บันทึกเมื่อ": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_customers(pd.concat([df, pd.DataFrame([row])], ignore_index=True))


def valid_phone(phone: str) -> bool:
    digits = re.sub(r"[\s-]", "", phone)
    return bool(re.fullmatch(r"0\d{8,9}", digits))


# ---------------------------------------------------------------
# แถบเมนูด้านซ้าย
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
<div class="brand">
    <div class="logo">📡</div>
    <div class="name">{APP_NAME}</div>
    <div class="sub">{APP_SUBTITLE}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio("เมนู", [PAGE_PREDICT, PAGE_CUSTOMER], label_visibility="collapsed")
    st.divider()
    st.caption(f"👥 ลูกค้าที่บันทึกไว้: **{len(load_customers())}** ราย")
    st.caption("สมาชิก: " + MEMBERS)


def hero(badge: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="hero">
    <div class="badge">{badge}</div>
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        f"""
<div class="footer">
    <b>สมาชิก:</b> {MEMBERS}<br>
    <span style="font-size:.82rem;">{APP_NAME} · โมเดลจากชุดข้อมูล Bank Marketing</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------
# หน้า 1: ทำนายโอกาส
# ---------------------------------------------------------------
def page_predict() -> None:
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}\n\nให้วาง bank_tree.joblib ไว้โฟลเดอร์เดียวกับ app.py")
        st.stop()

    hero("📡 AI · Binary Classification · Decision Tree", APP_NAME, APP_SUBTITLE)

    # เลือกลูกค้าที่บันทึกไว้ (ไม่บังคับ) เพื่อดึงอายุมาใส่ให้อัตโนมัติ
    customers = load_customers()
    NONE_OPT = "— ไม่ระบุลูกค้า (กรอกเอง) —"
    options = [NONE_OPT] + [f"{r['id']} · {r['ชื่อ-นามสกุล']}" for _, r in customers.iterrows()]

    st.session_state.setdefault("age", 40)

    def fill_from_customer():
        sel = st.session_state.get("pick_customer", NONE_OPT)
        if sel != NONE_OPT:
            cid = sel.split(" · ")[0]
            row = customers[customers["id"] == cid]
            if len(row) and str(row.iloc[0]["อายุ"]).isdigit():
                st.session_state["age"] = min(max(int(row.iloc[0]["อายุ"]), 18), 95)

    with st.container(border=True):
        st.markdown('<div class="section-title">👤 ข้อมูลลูกค้า</div>', unsafe_allow_html=True)
        picked = st.selectbox(
            "เลือกจากลูกค้าที่บันทึกไว้ (ไม่บังคับ)",
            options,
            key="pick_customer",
            on_change=fill_from_customer,
            help="ถ้าเลือก ระบบจะดึงอายุของลูกค้ามาใส่ให้ และแสดงชื่อในผลทำนาย",
        )
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("อายุ (ปี)", 18, 95, key="age")
        with c2:
            balance = st.number_input("ยอดเงินคงเหลือในบัญชี (balance)", -6847, 81204, 1000, step=100)

        c3, c4, c5 = st.columns(3)
        with c3:
            default = st.radio("มีหนี้ผิดนัดชำระ", ["ไม่มี", "มี"], horizontal=True)
        with c4:
            housing = st.radio("มีสินเชื่อบ้าน", ["ไม่มี", "มี"], horizontal=True)
        with c5:
            loan = st.radio("มีสินเชื่อส่วนบุคคล", ["ไม่มี", "มี"], horizontal=True)

    with st.container(border=True):
        st.markdown('<div class="section-title">📞 ประวัติการติดต่อ</div>', unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            duration = st.slider("ระยะเวลาโทรคุยครั้งล่าสุด (วินาที)", 2, 3881, 300, step=10)
            st.caption(f"≈ {duration // 60} นาที {duration % 60} วินาที")
        with d2:
            campaign = st.slider("จำนวนครั้งที่ติดต่อในแคมเปญนี้", 1, 63, 2)
        previous = st.slider("จำนวนครั้งที่เคยติดต่อก่อนแคมเปญนี้", 0, 58, 0)

    if st.button("🔍 ทำนายโอกาส", type="primary"):
        row = {
            "age": minmax("age", age),
            "balance": minmax("balance", balance),
            "duration": minmax("duration", duration),
            "campaign": minmax("campaign", campaign),
            "previous": minmax("previous", previous),
            "default": 1 if default == "มี" else 0,
            "housing": 1 if housing == "มี" else 0,
            "loan": 1 if loan == "มี" else 0,
        }
        X = pd.DataFrame([row])[list(model.feature_names_in_)]

        pred = int(model.predict(X)[0])
        p_yes = float(model.predict_proba(X)[0][list(model.classes_).index(1)])

        if pred == 1:
            css, verdict, tip = "result-yes", "✅ มีแนวโน้มเปิดบัญชีเงินฝากประจำ", "ควรให้ความสำคัญกับลูกค้ารายนี้ในการติดต่อต่อ"
        else:
            css, verdict, tip = "result-no", "⚠️ มีโอกาสเปิดบัญชีค่อนข้างต่ำ", "อาจพิจารณาจัดลำดับความสำคัญของการติดต่อลงมา"

        who = "ผลการทำนาย" if picked == NONE_OPT else f"ผลการทำนาย · ลูกค้า: {picked.split(' · ', 1)[1]}"

        st.markdown(
            f"""
<div class="result-card {css}">
    <div class="label">{who}</div>
    <div class="verdict">{verdict}</div>
    <div class="pct">{p_yes:.1%}</div>
    <div class="label">โอกาสที่ลูกค้าจะเปิดบัญชี</div>
    <div class="bar-bg"><div class="bar-fg" style="width:{p_yes * 100:.1f}%"></div></div>
    <div class="label" style="margin-top:12px;">💡 {tip}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.expander("ดูข้อมูลที่ส่งเข้าโมเดล (หลังสเกลแล้ว)"):
            st.dataframe(X)

    footer()


# ---------------------------------------------------------------
# หน้า 2: เก็บข้อมูลลูกค้า
# ---------------------------------------------------------------
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
    with s1:
        st.markdown(f'<div class="stat-box"><div class="n">{len(df)}</div><div class="t">ลูกค้าทั้งหมด</div></div>', unsafe_allow_html=True)
    with s2:
        avg = f"{ages.mean():.0f}" if len(df) and ages.notna().any() else "-"
        st.markdown(f'<div class="stat-box"><div class="n">{avg}</div><div class="t">อายุเฉลี่ย (ปี)</div></div>', unsafe_allow_html=True)
    with s3:
        fem = int((df["เพศ"] == "หญิง").sum())
        st.markdown(f'<div class="stat-box"><div class="n">{fem} / {len(df) - fem}</div><div class="t">หญิง / อื่น ๆ</div></div>', unsafe_allow_html=True)

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
                st.download_button(
                    "⬇️ ดาวน์โหลด CSV",
                    df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="customers.csv",
                    mime="text/csv",
                )
            with d2:
                with st.popover("🗑️ ลบข้อมูลลูกค้า"):
                    labels = {f"{r['id']} · {r['ชื่อ-นามสกุล']}": r["id"] for _, r in df.iterrows()}
                    to_delete = st.multiselect("เลือกรายการที่ต้องการลบ", list(labels))
                    if st.button("ยืนยันการลบ", disabled=not to_delete):
                        ids = {labels[k] for k in to_delete}
                        save_customers(df[~df["id"].isin(ids)])
                        st.session_state["flash"] = f"ลบข้อมูล {len(ids)} รายการแล้ว"
                        st.rerun()

    footer()


# ---------------------------------------------------------------
# เลือกหน้าตามเมนู
# ---------------------------------------------------------------
if page == PAGE_PREDICT:
    page_predict()
else:
    page_customer()
