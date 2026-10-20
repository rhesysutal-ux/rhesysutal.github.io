import hashlib
import io
import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

REF_DATE = pd.Timestamp("2026-01-01")      # tanggal acuan Recency (sehari setelah akhir data 2025)
UNK = "Tidak Diketahui"
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus",
         "September", "Oktober", "November", "Desember"]


# ============================================================ ETL / MODEL (tanpa UI)
def run_etl(data: bytes):
    xl = pd.ExcelFile(io.BytesIO(data))
    raw = xl.parse("Data_Raw")
    prod = xl.parse("Data_Produk")
    cus = xl.parse("Data_Pelanggan")
    ref = prod.set_index("Product_ID")["Reference_Unit_Price_Rp"]
    pid = prod.set_index("Product_Name")["Product_ID"]
    pcat = prod.set_index("Product_ID")["Category"]
    cm = cus.set_index("Customer_ID")
    R = {"raw": raw, "prod": prod, "cus": cus, "fid": hashlib.md5(data).hexdigest(),
         "dict": xl.parse("Data_Dictionary"), "info": xl.parse("Info_Dataset", header=None)}

    # ---------------------------------------------------------- profiling (sebelum)
    p = {"rows": raw.shape[0], "cols": raw.shape[1]}
    p["miss_col"] = raw.isna().sum()
    p["missing"] = int(p["miss_col"].sum())
    p["dup"] = int(raw.duplicated().sum())
    p["q_bad"] = int(((raw["Quantity"] <= 0) | (raw["Quantity"] > 50)).sum())
    rp_ = raw["Product_Name"].map(pid).map(ref)
    p["p_bad"] = int((raw["Unit_Price"].isna() | (raw["Unit_Price"] <= 0) | (raw["Unit_Price"] > 3 * rp_)).sum())
    ok = raw["Quantity"].notna() & raw["Unit_Price"].notna() & raw["Line_Total"].notna()
    p["lt_bad"] = int((ok & ((raw["Quantity"] * raw["Unit_Price"] * (1 - raw["Discount_Rate"]) - raw["Line_Total"]).abs() > 1)).sum())
    p["cat_txt"] = int((raw["Category"] != raw["Category"].str.strip().str.title().replace({"Atk": "ATK"})).sum())
    pm = raw["Payment_Method"].dropna()
    p["pay_txt"] = int((pm != pm.str.strip().replace({"Qris": "QRIS", "qris": "QRIS"})).sum())
    p["cat_n"], p["pay_n"] = raw["Category"].nunique(), raw["Payment_Method"].nunique()
    p["ntx"], p["ncus"], p["nprod"] = raw["Transaction_ID"].nunique(), raw["Customer_ID"].nunique(), raw["Product_ID"].nunique()
    p["sales"] = float(raw["Line_Total"].sum())
    p["dtypes"] = raw.dtypes.astype(str)
    R["p"] = p

    # ---------------------------------------------------------- cleaning
    d = raw.drop_duplicates().copy()
    n_dup = len(raw) - len(d)
    d["Category"] = d["Category"].str.strip().str.title().replace({"Atk": "ATK"})
    d["Payment_Method"] = d["Payment_Method"].str.strip().replace({"Qris": "QRIS", "qris": "QRIS"})
    c_fill = int(d["Customer_ID"].isna().sum())
    d["Customer_ID"] = d["Customer_ID"].fillna(d.groupby("Transaction_ID")["Customer_ID"].transform("first"))
    c_left = int(d["Customer_ID"].isna().sum())
    d["Product_ID"] = d["Product_ID"].fillna(d["Product_Name"].map(pid))
    d["Category"] = d["Product_ID"].map(pcat)
    m_reg, m_typ = int(d["Region"].isna().sum()), int(d["Customer_Type"].isna().sum())
    d["Region"] = d["Region"].fillna(d["Customer_ID"].map(cm["Region"])).fillna(UNK)
    d["Customer_Type"] = d["Customer_Type"].fillna(d["Customer_ID"].map(cm["Customer_Type"])).fillna(UNK)
    m_pay = int(d["Payment_Method"].isna().sum())
    d["Payment_Method"] = d["Payment_Method"].fillna(d.groupby("Transaction_ID")["Payment_Method"].transform("first")).fillna(UNK)
    rpx = d["Product_ID"].map(ref)
    bp = d["Unit_Price"].isna() | (d["Unit_Price"] <= 0) | (d["Unit_Price"] > 3 * rpx)
    d.loc[bp, "Unit_Price"] = rpx[bp]
    der = (d["Line_Total"] / (d["Unit_Price"] * (1 - d["Discount_Rate"]))).round()
    bq = d["Quantity"].isna() | (d["Quantity"] <= 0) | (d["Quantity"] > 50)
    okq = bq & der.between(1, 6)
    d.loc[okq, "Quantity"] = der[okq]
    med = d.loc[~bq, "Quantity"].median()
    d.loc[bq & ~okq, "Quantity"] = med
    d["Quantity"] = d["Quantity"].astype(int)
    d["Unit_Price"] = d["Unit_Price"].astype(int)
    d["Line_Total"] = (d["Quantity"] * d["Unit_Price"] * (1 - d["Discount_Rate"])).round(0).astype(int)
    d["Transaction_Date"] = pd.to_datetime(d["Transaction_Date"])
    d["Bulan"] = d["Transaction_Date"].dt.month
    n_pre = len(d)
    d = d.drop_duplicates().sort_values(["Transaction_ID", "Product_ID"]).reset_index(drop=True)   # duplikat yang baru terlihat setelah perbaikan angka
    n_dup += n_pre - len(d)
    R["clean"] = d
    R["log"] = pd.DataFrame([
        ["Baris duplikat persis", n_dup, "Dihapus (drop_duplicates)", "Duplikat menggandakan penjualan dan menggeser support Apriori."],
        ["Customer_ID kosong", c_fill, "Diisi dari baris lain pada Transaction_ID yang sama (sisa: %d)" % c_left, "Satu nota hanya milik satu pelanggan."],
        ["Region kosong", m_reg, "Dilengkapi dari Data_Pelanggan; sisa = '%s'" % UNK, "Master pelanggan adalah sumber valid."],
        ["Customer_Type kosong", m_typ, "Dilengkapi dari Data_Pelanggan; sisa = '%s'" % UNK, "Master pelanggan adalah sumber valid."],
        ["Product_ID kosong", int(raw["Product_ID"].isna().sum()), "Dipetakan dari Product_Name (Data_Produk)", "Nama produk 1:1 dengan ID."],
        ["Payment_Method kosong", m_pay, "Diisi dari baris lain pada nota yang sama / '%s'" % UNK, "Pembayaran dilakukan per nota."],
        ["Teks tidak konsisten (Category, Payment)", p["cat_txt"] + p["pay_txt"], "Strip spasi, seragamkan huruf; Category diambil dari Data_Produk",
         "Mencegah satu kategori/metode terhitung ganda."],
        ["Unit_Price tidak valid", int(bp.sum()), "Diganti harga referensi Data_Produk", "Harga transaksi normal sama dengan harga referensi."],
        ["Quantity tidak valid (<=0, 999, kosong)", int(bq.sum()), "Line_Total / (Harga x (1 - Diskon)); cadangan: median",
         "Mempertahankan baris dan nilai yang tercatat; bukan pembelian grosir."],
        ["Line_Total tidak konsisten", p["lt_bad"], "Dihitung ulang = Qty x Harga x (1 - Diskon)", "Menjamin konsistensi total."],
    ], columns=["Masalah", "Jumlah", "Penanganan", "Alasan"])
    R["n"] = dict(dup=n_dup, c_fill=c_fill, c_left=c_left, bp=int(bp.sum()), bq=int(bq.sum()), okq=int(okq.sum()))
    R["ba"] = pd.DataFrame([
        ["Jumlah baris", p["rows"], len(d)], ["Sel kosong", p["missing"], int(d.drop(columns="Bulan").isna().sum().sum())],
        ["Baris duplikat", p["dup"], int(d.drop(columns="Bulan").duplicated().sum())],
        ["Kategori unik", p["cat_n"], d["Category"].nunique()], ["Variasi metode bayar", p["pay_n"], d["Payment_Method"].nunique()],
        ["Quantity tidak wajar", p["q_bad"], int(((d["Quantity"] <= 0) | (d["Quantity"] > 50)).sum())],
        ["Total tidak konsisten", p["lt_bad"], 0], ["Total penjualan (Rp)", int(p["sales"]), int(d["Line_Total"].sum())],
        ["Transaksi unik", p["ntx"], d["Transaction_ID"].nunique()], ["Pelanggan unik", p["ncus"], d["Customer_ID"].nunique()],
    ], columns=["Indikator", "Sebelum", "Sesudah"])

    # ---------------------------------------------------------- fitur
    tx = d.groupby("Transaction_ID").agg(Tanggal=("Transaction_Date", "first"), Customer_ID=("Customer_ID", "first"),
                                          basket_size=("Product_ID", "nunique"), Total=("Line_Total", "sum"))
    R["tx"] = tx
    g = tx.dropna(subset=["Customer_ID"]).groupby("Customer_ID")
    rfm = pd.DataFrame({"Recency": (REF_DATE - g["Tanggal"].max()).dt.days, "Frequency": g.size(), "Monetary": g["Total"].sum()}).reset_index()
    rfm["Log_Recency"] = np.log1p(rfm["Recency"])
    rfm["Log_Frequency"] = np.log1p(rfm["Frequency"])
    rfm["Log_Monetary"] = np.log1p(rfm["Monetary"])
    for c in ["Log_Recency", "Log_Frequency", "Log_Monetary"]:
        rfm["Z_" + c[4:]] = (rfm[c] - rfm[c].mean()) / rfm[c].std(ddof=0)
    R["rfm"] = rfm
    X = rfm[["Z_Recency", "Z_Frequency", "Z_Monetary"]].values
    el = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=42).fit(X)
        el.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    R["elbow"] = pd.DataFrame(el)
    return R


def apriori(df, minsup, minconf, maxlen=3):
    """Apriori manual pada matriks boolean transaksi x produk. Return (n, itemsets, rules)."""
    B = pd.crosstab(df["Transaction_ID"], df["Product_Name"]).clip(upper=1).astype(bool)
    n, cols, A = len(B), list(B.columns), B.values
    if n == 0:
        return 0, {}, pd.DataFrame(columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
    fs = {(i,): A[:, i].mean() for i in range(len(cols)) if A[:, i].mean() >= minsup}
    f1 = [k[0] for k in fs]
    for a, b in itertools.combinations(f1, 2):
        s = (A[:, a] & A[:, b]).mean()
        if s >= minsup:
            fs[(a, b)] = s
    if maxlen >= 3:
        for a, b, c in itertools.combinations(f1, 3):
            if (a, b) in fs and (a, c) in fs and (b, c) in fs:
                s = (A[:, a] & A[:, b] & A[:, c]).mean()
                if s >= minsup:
                    fs[(a, b, c)] = s
    rules = []
    for it, s in fs.items():
        if len(it) < 2:
            continue
        for k in range(1, len(it)):
            for ant in itertools.combinations(it, k):
                cn = tuple(i for i in it if i not in ant)
                conf = s / fs[ant]
                if conf >= minconf:
                    rules.append([" + ".join(cols[i] for i in ant), " + ".join(cols[i] for i in cn), s, conf, conf / fs[cn]])
    rules = pd.DataFrame(rules, columns=["Antecedent", "Consequent", "Support", "Confidence", "Lift"])
    rules = rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)
    isets = pd.DataFrame([[" + ".join(cols[i] for i in k), len(k), v, int(round(v * n))] for k, v in fs.items()],
                         columns=["Itemset", "Ukuran", "Support", "Jumlah Transaksi"]).sort_values("Support", ascending=False)
    return n, isets, rules


def seg_names(k):
    base = {2: ["Loyal Customers", "Churn Risk"], 3: ["Loyal Customers", "Potential Customers", "Churn Risk"],
            4: ["Loyal Customers", "Potential Customers", "New / Low-Value Customers", "Churn Risk"]}
    return base.get(k) or ["Loyal Customers"] + ["Potential Customers %d" % i for i in range(1, k - 1)] + ["Churn Risk"]


def segment(rfm, k):
    X = rfm[["Z_Recency", "Z_Frequency", "Z_Monetary"]].values
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    out = rfm.copy()
    out["_c"] = km.labels_
    z = out.groupby("_c")[["Z_Recency", "Z_Frequency", "Z_Monetary"]].mean()
    order = (-z["Z_Recency"] + z["Z_Frequency"] + z["Z_Monetary"]).sort_values(ascending=False).index
    names = seg_names(k)
    lab = {c: names[i] for i, c in enumerate(order)}
    out["Segmen"] = out["_c"].map(lab)
    out["Cluster"] = out["_c"].map({c: i for i, c in enumerate(order)})
    out = out.drop(columns="_c")
    prof = out.groupby("Segmen").agg(Pelanggan=("Customer_ID", "count"), Recency=("Recency", "mean"), Frequency=("Frequency", "mean"),
                                     Monetary=("Monetary", "mean"), Total_Monetary=("Monetary", "sum")).reindex(names).reset_index()
    prof["% Pelanggan"] = prof["Pelanggan"] / prof["Pelanggan"].sum() * 100
    prof["% Revenue"] = prof["Total_Monetary"] / prof["Total_Monetary"].sum() * 100
    return out, prof, int(km.n_iter_), float(km.inertia_)


KARAKTER = {"Loyal": "Belanja baru-baru ini, sering, dan bernilai besar. Tulang punggung revenue.",
            "Potential": "Cukup bernilai namun Recency mulai memburuk / frekuensi menengah. Mudah ditingkatkan menjadi Loyal.",
            "New": "Aktif baru-baru ini tetapi Frequency dan Monetary rendah.",
            "Churn": "Lama tidak bertransaksi, Frequency dan Monetary rendah. Berisiko berhenti."}
STRATEGI = {"Loyal": "Program loyalitas/poin eksklusif, akses awal produk baru, bundling premium; hindari diskon besar.",
            "Potential": "Voucher belanja berikutnya dan rekomendasi cross-sell berbasis Apriori untuk menaikkan frekuensi.",
            "New": "Paket sambutan, edukasi produk kebutuhan pokok, insentif kunjungan ulang < 30 hari.",
            "Churn": "Kampanye reaktivasi (diskon personal, WhatsApp), survei alasan berhenti; anggaran dibatasi."}


def kind(name):
    return next(k for k in ("Loyal", "Potential", "New", "Churn") if name.startswith(k))


# ============================================================ UI (STREAMLIT)
import base64, html, os
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BG, ACCENT, TEXT = "#800000", "#E63946", "#2B0A10"
NAMA, NPM = "Maria Restilia Sutal", "51230066"                       # ganti sesuai identitas
UNIVERSITAS, FAKULTAS = "UNIVERSITAS TIMOR", "Fakultas Pertanian, Sains, dan Kesehatan"
PRODI, MATA_KULIAH = "Program Studi Teknologi Informasi", "Business Intelligence"
LOGO_FILE, FOTO_FILE = ["logo.png", "logo.jpg", "logo.jpeg"], ["foto.jpg", "foto.jpeg", "foto.png"]
# Foto profil (tertanam di kode; jika ada file foto.jpg/foto.png di folder yang sama, file itu yang dipakai)
FOTO_PROFIL = "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"
EXCEL_DEFAULT = "P8_UTS_Business_Intelligence.xlsx"
PALETTE = ["#E63946", "#1D8FE1", "#2A9D8F", "#F4A261", "#8E44AD", "#F1C40F", "#2ECC71", "#FF6FB5", "#264653", "#E76F51"]
SEG_COLOR = {"Loyal Customers": "#2A9D8F", "Potential Customers": "#1D8FE1", "New / Low-Value Customers": "#F4A261", "Churn Risk": "#E63946"}
KETERANGAN = ("Data transaksi toko (18.513 baris item) dieksplorasi, dibersihkan, diubah menjadi basket transaksi dan fitur RFM, "
              "dianalisis dengan Apriori (pola kombinasi produk) dan K-Means (segmentasi pelanggan), disajikan dalam dashboard BI, "
              "lalu diterjemahkan menjadi rekomendasi keputusan bisnis.")
MENU = ["Ringkasan", "Tahap 1 · Business Understanding & EDA", "Tahap 2 · Data Preprocessing", "Tahap 3 · Analisis Apriori",
        "Tahap 4 · Segmentasi K-Means", "Tahap 5 · BI Dashboard", "Tahap 6 · Rekomendasi Bisnis", "Luaran & Export"]

CSS = """<style>
.stApp,[data-testid="stHeader"]{background-color:#800000;}
[data-testid="stSidebar"]{background-color:#4A0E17;}
[data-testid="stMetric"]{background:#FFFFFF;border-left:4px solid #E63946;border-radius:10px;padding:12px 16px;}
h1,h2,h3,h4,p,label,li,[data-testid="stMarkdownContainer"],[data-testid="stCaptionContainer"],[data-testid="stWidgetLabel"]{color:#FFFFFF !important;}
[data-testid="stMetric"] *,[data-testid="stExpander"] *,button *{color:#2B0A10 !important;}
[data-testid="stExpander"]{background:#FFFFFF;border-radius:10px;}
[data-testid="stAlert"] *{color:#2B0A10 !important;}
[data-testid="stSidebar"] *{color:#FFFFFF !important;}
[data-testid="stSidebar"] button *,[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *{color:#2B0A10 !important;}
[data-testid="stSidebar"] div[role="radiogroup"] label{padding:8px 12px;border-radius:8px;margin-bottom:2px;width:100%;}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{background:#7A1F2E;}
.rumus-box{background:#FFFFFF;border:2px solid #E63946;border-radius:12px;padding:16px 20px;margin:14px 0 22px 0;}
.rumus-title{color:#800000;font-weight:700;font-size:1.05rem;margin-bottom:10px;}
.rumus-item{border-top:1px dashed #E0A3A9;padding:10px 0;}.rumus-item:first-of-type{border-top:none;}
.rumus-name{color:#2B0A10;font-weight:600;margin-bottom:6px;}
.rumus-formula{display:block;background:#FCEBEC;color:#6B0F1A;padding:10px 14px;border-radius:8px;font-family:Consolas,'Courier New',monospace;font-size:.95rem;white-space:pre-wrap;}
.rumus-note{color:#4A2A2F;margin-top:8px;font-size:.92rem;line-height:1.5;}
.note-box{background:#FFFFFF;border-left:4px solid #E63946;border-radius:10px;padding:14px 18px;margin:12px 0 18px 0;color:#2B0A10 !important;line-height:1.55;}
.note-box b{color:#800000;}
.soal{background:#FCEBEC;border-left:6px solid #E63946;border-radius:10px;padding:10px 16px;margin:22px 0 8px 0;color:#800000 !important;font-weight:800;font-size:1.15rem;}
.seg-card{background:#FFFFFF;border-radius:12px;padding:14px 18px;margin-bottom:12px;border-top:4px solid #E63946;color:#2B0A10;}
.seg-card .t{font-weight:800;font-size:1.05rem;margin-bottom:6px;color:#2B0A10 !important;}
.seg-card .d{color:#4A2A2F !important;font-size:.92rem;line-height:1.5;}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 18px 0;}
.kpi{background:#FFFFFF;border-left:4px solid #E63946;border-radius:10px;padding:12px 16px;}
.kpi-label{color:#4A2A2F !important;font-size:.88rem;margin:0 0 8px 0;min-height:2.4em;}
.kpi-value{color:#800000 !important;font-size:1.7rem;font-weight:800;}
.cover{background:#FFFFFF;border-top:10px solid #E63946;border-radius:18px;padding:34px 40px 36px;text-align:center;box-shadow:0 14px 40px rgba(0,0,0,.35);margin-top:1.5rem;}
.cover-logo{width:92px;height:92px;border-radius:50%;background:#FCEBEC;border:3px solid #E63946;margin:0 auto 12px;display:flex;align-items:center;justify-content:center;font-size:2.6rem;overflow:hidden;}
.cover-logo img{width:100%;height:100%;object-fit:contain;}
.cover-univ{color:#800000 !important;font-size:1.5rem;font-weight:800;line-height:1.25;}
.cover-fak,.cover-mk{color:#6B0F1A !important;font-size:1rem;margin-top:4px;}
.cover-line{height:3px;width:80px;background:#E63946;border-radius:3px;margin:22px auto;}
.cover-title{color:#2B0A10 !important;font-size:1.9rem;font-weight:800;line-height:1.2;margin:8px 0 4px;}
.cover-sub{color:#4A2A2F !important;font-size:1rem;}
.cover-foto{width:150px;height:150px;border-radius:50%;object-fit:cover;border:4px solid #E63946;margin:26px auto 12px;display:flex;align-items:center;justify-content:center;font-size:3.6rem;background:#FCEBEC;}
.cover-nama{color:#2B0A10 !important;font-size:1.3rem;font-weight:700;}.cover-npm{color:#6B0F1A !important;font-size:1.05rem;margin-top:2px;}
</style>"""


def rp(x): return "Rp " + f"{x:,.0f}".replace(",", ".")
def num(x): return f"{x:,.0f}".replace(",", ".")
def pct(x, d=1): return f"{x * 100:.{d}f}%".replace(".", ",")
def f2(x): return f"{x:.2f}".replace(".", ",")


def soal(t): st.markdown(f'<div class="soal">{html.escape(t)}</div>', unsafe_allow_html=True)
def note(t): st.markdown(f'<div class="note-box">{t}</div>', unsafe_allow_html=True)


def kpi(items):
    cells = "".join(f'<div class="kpi"><div class="kpi-label">{html.escape(l)}</div><div class="kpi-value">{v}</div></div>' for l, v in items)
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def rumus_box(items, judul="📐 Rumus & Penjelasan"):
    s = [f'<div class="rumus-box"><div class="rumus-title">{html.escape(judul)}</div>']
    for n, f, e in items:
        s.append(f'<div class="rumus-item"><div class="rumus-name">{html.escape(n)}</div><code class="rumus-formula">{html.escape(f)}</code>'
                 f'<div class="rumus-note"><b>Penjelasan:</b> {html.escape(e)}</div></div>')
    st.markdown("".join(s) + "</div>", unsafe_allow_html=True)


def jawaban(key, pertanyaan, teks, h=150):
    st.markdown(f"**❓ {pertanyaan}**")
    st.text_area("Jawaban / Catatan Mahasiswa", value=teks, key=key, height=h)


def show_df(d, **kw):
    try: st.dataframe(d, width="stretch", hide_index=True, **kw)
    except Exception: st.dataframe(d, use_container_width=True, hide_index=True, **kw)


def show_fig(fig, h=360):
    fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=h, colorway=PALETTE,
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(color=TEXT))
    try: st.plotly_chart(fig, width="stretch")
    except Exception: st.plotly_chart(fig, use_container_width=True)


def fmt_rules(r):
    r = r.copy()
    r.insert(0, "No", range(1, len(r) + 1))
    r["Support"], r["Confidence"], r["Lift"] = r["Support"].map(lambda v: pct(v, 2)), r["Confidence"].map(pct), r["Lift"].map(f2)
    return r


def network(rules, h=480):
    ns = list(dict.fromkeys(list(rules["Antecedent"]) + list(rules["Consequent"])))
    pos = {n: (np.cos(2 * np.pi * i / len(ns)), np.sin(2 * np.pi * i / len(ns))) for i, n in enumerate(ns)}
    fig, mx = go.Figure(), rules["Lift"].max()
    for r in rules.itertuples():
        (x0, y0), (x1, y1) = pos[r.Antecedent], pos[r.Consequent]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=1 + 5 * r.Lift / mx, color="#F4A261"),
                                 hoverinfo="text", text=f"{r.Antecedent} → {r.Consequent}<br>Lift {r.Lift:.2f}", showlegend=False))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns], mode="markers+text", text=ns, textposition="top center",
                             marker=dict(size=16, color="#800000"), showlegend=False))
    fig.update_xaxes(visible=False, range=[-1.5, 1.5]); fig.update_yaxes(visible=False, range=[-1.3, 1.3])
    fig.update_layout(title="Network Graph Aturan Asosiasi (ketebalan garis = lift)")
    show_fig(fig, h)


@st.cache_data(show_spinner="Menjalankan EDA, preprocessing, RFM & Elbow...")
def cached_etl(data: bytes): return run_etl(data)


@st.cache_data(show_spinner="Menjalankan Apriori...")
def cached_apriori(_df, key, ms, mc): return apriori(_df, ms, mc)


@st.cache_data(show_spinner="Menjalankan K-Means...")
def cached_segment(_rfm, fid, k): return segment(_rfm, k)


def seg_data(R): return cached_segment(R["rfm"], R["fid"], st.session_state.get("k_final", 3))


def bonus_rules(R, seg):
    d = R["clean"].merge(seg[["Customer_ID", "Segmen"]], on="Customer_ID", how="left")
    out = []
    for s in seg["Segmen"].unique():
        _, _, r = apriori(d[d["Segmen"] == s], 0.02, 0.2, 2)
        r = r[r["Lift"] > 1].head(3).copy(); r.insert(0, "Segmen", s); out.append(r)
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------ RINGKASAN
def page_ringkasan(R):
    p = R["p"]
    st.header("RINGKASAN")
    st.subheader("ALUR ANALISIS (sesuai lembar soal)")
    nodes = [("Tahap 1", "Business Understanding & EDA", "10%", "#E63946"), ("Tahap 2", "Data Preprocessing", "25%", "#F4A261"),
             ("Tahap 3", "Apriori", "20%", "#F1C40F"), ("Tahap 4", "K-Means", "20%", "#2A9D8F"),
             ("Tahap 5", "BI Dashboard", "15%", "#1D8FE1"), ("Tahap 6", "Rekomendasi", "10%", "#B07CC6")]
    dot = 'digraph G {graph [rankdir=LR,bgcolor="transparent",nodesep=0.3,ranksep=0.4];edge [penwidth=3,color="#FFFFFF"];'
    for i, (a, b, w, c) in enumerate(nodes):
        dot += f'N{i} [shape=box,style="rounded,filled",fillcolor="{c}",color="#FFFFFF",penwidth=2,label=<<B>{a}</B><BR/>{b}<BR/>{w}>,fontcolor="#1F2D33"];'
    dot += " -> ".join(f"N{i}" for i in range(6)) + ";}"
    st.graphviz_chart(dot, use_container_width=True)
    kpi([("Baris data mentah", num(p["rows"])), ("Kolom", p["cols"]), ("Transaksi unik", num(p["ntx"])),
         ("Pelanggan unik", num(p["ncus"])), ("Produk", p["nprod"]), ("Baris bersih", num(len(R["clean"])))])
    note(f"<b>Keterangan</b><br>{html.escape(KETERANGAN)}")
    st.subheader("Rubrik penilaian")
    show_df(pd.DataFrame({"No": range(1, 7), "Komponen": ["Business Understanding & EDA", "Data Preprocessing", "Analisis Apriori",
                                                        "Segmentasi K-Means", "Dashboard Business Intelligence", "Rekomendasi Keputusan Bisnis"],
                          "Bobot": ["10%", "25%", "20%", "20%", "15%", "10%"]}))
    rumus_box([("Support", "Support(A→B) = transaksi(A dan B) / total transaksi", "Seberapa sering kombinasi muncul."),
               ("Confidence", "Confidence(A→B) = Support(A,B) / Support(A)", "Peluang B dibeli jika A dibeli."),
               ("Lift", "Lift = Confidence(A→B) / Support(B)", "Lift > 1 berarti asosiasi positif."),
               ("RFM", "R = 1 Jan 2026 − transaksi terakhir;  F = jumlah transaksi;  M = Σ Line_Total", "Dasar segmentasi K-Means."),
               ("Standardisasi", "Z = (x − mean) / std", "Menyetarakan skala fitur K-Means.")])


# ------------------------------------------------------------------ TAHAP 1
def page_t1(R):
    p, raw, d = R["p"], R["raw"], R["clean"]
    st.header("Tahap 1 — Business Understanding & EDA (10%)")
    soal("Soal 1. Memahami Permasalahan Bisnis")
    jawaban("s1a", "a. Identifikasi permasalahan utama perusahaan.",
            "Data transaksi hanya dipakai untuk laporan rutin. Perusahaan belum mengetahui pola pembelian, kombinasi produk yang dibeli bersamaan, "
            "dan segmen pelanggan, sedangkan kualitas data masih bermasalah (missing values, duplikat, outlier, format tidak konsisten).", 120)
    jawaban("s1b", "b. Rumuskan minimal tiga pertanyaan bisnis.",
            "1) Kombinasi produk apa yang sering dibeli bersamaan dan layak di-bundling?\n2) Bagaimana segmen pelanggan berdasarkan RFM dan siapa yang berisiko churn?\n"
            "3) Produk, kategori, dan bulan apa yang menyumbang penjualan terbesar?\n4) Berapa rata-rata basket size dan bagaimana tren penjualan 2025?", 140)
    jawaban("s1c", "c. Bagaimana analisis membantu meningkatkan penjualan dan mempertahankan pelanggan?",
            "Apriori mendukung cross-selling, bundling, dan tata letak rak untuk menaikkan nilai transaksi. K-Means mendukung program retensi tepat sasaran "
            "sehingga pelanggan bernilai tinggi dipertahankan dan pelanggan berisiko churn dapat diaktifkan kembali.", 120)

    soal("Soal 2. Eksplorasi Data")
    st.markdown("**a. Dimensi, tipe data, dan rentang waktu**")
    kpi([("Jumlah baris", num(p["rows"])), ("Jumlah kolom", p["cols"]),
         ("Periode", f"{raw['Transaction_Date'].min():%d %b %Y} – {raw['Transaction_Date'].max():%d %b %Y}")])
    show_df(pd.DataFrame({"Kolom": p["dtypes"].index, "Tipe data": p["dtypes"].values, "Missing": p["miss_col"].values}))
    st.markdown("**b. Statistik deskriptif**")
    kpi([("Transaksi unik", num(p["ntx"])), ("Pelanggan unik", num(p["ncus"])), ("Produk unik", p["nprod"]),
         ("Rata-rata Quantity (wajar)", f2(raw.loc[(raw["Quantity"] > 0) & (raw["Quantity"] < 50), "Quantity"].mean())),
         ("Total penjualan (mentah)", rp(p["sales"]))])
    show_df(raw[["Quantity", "Unit_Price", "Discount_Rate", "Line_Total"]].describe().round(2).reset_index())
    st.markdown("**c. Missing values, duplikat, dan outlier**")
    show_df(pd.DataFrame([["Baris duplikat persis", p["dup"]], ["Total sel kosong", p["missing"]], ["Quantity ≤ 0 atau > 50", p["q_bad"]],
                          ["Unit_Price kosong/≤0/jauh di atas referensi", p["p_bad"]], ["Line_Total ≠ Qty × Harga × (1 − Diskon)", p["lt_bad"]],
                          ["Teks tidak konsisten (Category, Payment)", p["cat_txt"] + p["pay_txt"]]], columns=["Masalah", "Jumlah baris"]))
    c = st.columns(2)
    with c[0]: show_fig(px.bar(x=p["miss_col"].index, y=p["miss_col"].values, title="Missing value per kolom", labels={"x": "Kolom", "y": "Jumlah"}), 320)
    with c[1]: show_fig(px.histogram(raw[raw["Quantity"].notna()], x="Quantity", nbins=40, title="Distribusi Quantity (terlihat outlier 999 / negatif)"), 320)
    st.markdown("**d. Visualisasi awal**")
    raw2 = raw.assign(Bulan=raw["Transaction_Date"].dt.month)
    m = raw2.groupby("Bulan", as_index=False)["Line_Total"].sum(); m["Bulan"] = m["Bulan"].map(lambda i: BULAN[i - 1])
    t = raw2.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index()
    c = st.columns(2)
    with c[0]: show_fig(px.line(m, x="Bulan", y="Line_Total", markers=True, title="Tren penjualan bulanan (data mentah)"), 360)
    with c[1]: show_fig(px.bar(t, x="Line_Total", y="Product_Name", orientation="h", title="10 produk penjualan tertinggi").update_yaxes(autorange="reversed"), 360)
    mx = m.loc[m["Line_Total"].idxmax()]; s3 = t["Line_Total"].head(3).sum() / p["sales"]
    jawaban("s2e", "e. Tuliskan minimal tiga temuan awal.",
            f"1) Penjualan tertinggi pada bulan {mx['Bulan']} ({rp(mx['Line_Total'])}); tren relatif stabil antar bulan.\n"
            f"2) Tiga produk teratas ({', '.join(t['Product_Name'].head(3))}) menyumbang {pct(s3)} penjualan; pendapatan terkonsentrasi pada produk bernilai tinggi.\n"
            f"3) Data belum siap: {num(p['missing'])} sel kosong, {p['dup']} duplikat, {p['q_bad']} Quantity tidak wajar, {p['cat_txt'] + p['pay_txt']} teks tidak konsisten.", 170)


# ------------------------------------------------------------------ TAHAP 2
def page_t2(R):
    p, n, d, rfm, tx = R["p"], R["n"], R["clean"], R["rfm"], R["tx"]
    st.header("Tahap 2 — Data Preprocessing (25%)")
    soal("Soal 3. Pembersihan dan Transformasi Data")
    st.markdown("**a & b. Penanganan masalah data dan alasannya**")
    show_df(R["log"])
    rumus_box([("Quantity tidak valid", "Quantity = Line_Total / (Unit_Price × (1 − Discount_Rate))", "Dihitung ulang dari total yang tercatat; cadangan: median."),
               ("Unit_Price tidak valid", "Unit_Price = Reference_Unit_Price_Rp (Data_Produk)", "Harga transaksi normal selalu sama dengan harga referensi."),
               ("Line_Total", "Line_Total = Quantity × Unit_Price × (1 − Discount_Rate)", "Konsistensi total setiap baris.")])
    st.markdown("**c. Transformasi format**")
    st.markdown("- Transaction_Date → datetime; diturunkan kolom **Bulan**.\n- Quantity, Unit_Price, Line_Total → integer; Discount_Rate → desimal (0,10 = 10%).\n"
                "- Category dan Payment_Method → kategori baku; Product_ID/Customer_ID tanpa spasi.")
    st.markdown("**d. Nilai ekstrem**")
    note(f"Quantity −2, 0, dan 999 dianggap kesalahan input (kuantitas wajar 1–6 per baris), bukan pembelian grosir, sehingga <b>dikoreksi, bukan dihapus</b> "
         f"agar transaksi tetap lengkap. Setelah koreksi Quantity maksimum = {d['Quantity'].max()} dan harga maksimum = {rp(d['Unit_Price'].max())}.")
    c = st.columns(2)
    with c[0]: show_fig(px.box(R["raw"], y="Quantity", title="Quantity sebelum"), 300)
    with c[1]: show_fig(px.box(d, y="Quantity", title="Quantity sesudah"), 300)
    st.markdown("**e. Perbandingan sebelum dan sesudah preprocessing**")
    show_df(R["ba"].astype(str))

    soal("Soal 4. Feature Engineering dan Persiapan Analisis")
    st.markdown("**a. Fitur RFM**")
    rumus_box([("RFM", "Recency = 1 Jan 2026 − transaksi terakhir | Frequency = COUNT(Transaction_ID unik) | Monetary = Σ Total transaksi",
                "Dihitung per pelanggan dari data bersih.")])
    show_df(rfm[["Customer_ID", "Recency", "Frequency", "Monetary"]])
    st.markdown("**b. basket_size (jumlah produk berbeda per transaksi)**")
    kpi([("Rata-rata basket_size", f2(tx["basket_size"].mean())), ("Maksimum", int(tx["basket_size"].max()))])
    bd = tx["basket_size"].value_counts().sort_index().reset_index(); bd.columns = ["basket_size", "Jumlah transaksi"]
    show_fig(px.bar(bd, x="basket_size", y="Jumlah transaksi", title="Distribusi basket_size"), 300)
    st.markdown("**c. Format data Apriori**")
    B = pd.crosstab(d["Transaction_ID"], d["Product_Name"]).clip(upper=1).astype(int)
    st.caption(f"Matriks transaksi {B.shape[0]} × {B.shape[1]} (1 = produk ada di transaksi). Kuantitas diabaikan karena yang dianalisis adalah keberadaan produk.")
    show_df(B.head(15).reset_index())
    st.markdown("**d & e. Matriks fitur K-Means dan standardisasi**")
    show_df(rfm[["Customer_ID", "Log_Recency", "Log_Frequency", "Log_Monetary", "Z_Recency", "Z_Frequency", "Z_Monetary"]].round(4))
    note("Fitur ditransformasi <b>log(1+x)</b> untuk meredam kemiringan distribusi, lalu <b>distandardisasi (Z-score)</b>. K-Means berbasis jarak Euclidean "
         "sehingga fitur berskala besar (Monetary jutaan rupiah) akan mendominasi tanpa penyetaraan skala.")
    c = st.columns(2)
    with c[0]: show_fig(px.histogram(rfm, x="Monetary", nbins=30, title="Monetary asli (miring kanan)"), 280)
    with c[1]: show_fig(px.histogram(rfm, x="Log_Monetary", nbins=30, title="Log_Monetary (lebih simetris)"), 280)
    st.markdown("**f. Strategi dataset besar**")
    st.markdown("- **Sampling**: uji parameter pada sampel acak, validasi pada data penuh.\n- **Chunking**: baca dan bersihkan data per potongan baris.\n"
                "- **Agregasi**: ringkas baris item menjadi tabel transaksi dan tabel pelanggan sebelum pemodelan.")
    st.markdown("**g. Dokumentasi jumlah data**")
    show_df(pd.DataFrame([["Dipertahankan", len(d), "Baris setelah deduplikasi"], ["Dihapus", n["dup"], "Duplikat persis"],
                          ["Diubah / dilengkapi", int(R["log"]["Jumlah"].iloc[1:].sum()), "Nilai kosong, harga, kuantitas, total, teks (hitungan per sel)"]],
                         columns=["Status", "Jumlah baris", "Keterangan"]))


# ------------------------------------------------------------------ TAHAP 3
def page_t3(R):
    d = R["clean"]
    st.header("Tahap 3 — Analisis Pola Pembelian Menggunakan Apriori (20%)")
    soal("Soal 5. Market Basket Analysis")
    st.markdown("**a & b. Matriks transaksi, minimum support, dan minimum confidence**")
    c = st.columns(2)
    ms = c[0].slider("Minimum support", 0.002, 0.05, 0.01, 0.002, format="%.3f")
    mc = c[1].slider("Minimum confidence", 0.05, 0.8, 0.2, 0.05)
    n, isets, rules = cached_apriori(d, (R["fid"], "all"), ms, mc)
    note(f"Rata-rata basket kecil ({f2(R['tx']['basket_size'].mean())} produk) dengan 30 produk, sehingga support kombinasi wajar kecil. Support {pct(ms)} "
         f"(≈{int(n * ms)} transaksi) cukup untuk menghindari pola kebetulan; confidence {pct(mc, 0)} menjaga aturan tetap relevan. Hanya aturan dengan lift > 1 dipertimbangkan.")
    st.markdown("**c. Frequent itemsets**")
    kpi([("Transaksi", num(n)), ("1-itemset", int((isets["Ukuran"] == 1).sum())), ("2-itemset", int((isets["Ukuran"] == 2).sum())),
         ("3-itemset", int((isets["Ukuran"] == 3).sum()))])
    x = isets[isets["Ukuran"] >= 2].head(15).copy(); x["Support"] = x["Support"].map(lambda v: pct(v, 2)); show_df(x)
    st.markdown("**d & e. Aturan asosiasi (support, confidence, lift)**")
    rules = rules[rules["Lift"] > 1]
    st.caption(f"{len(rules)} aturan dengan lift > 1 memenuhi batas minimum.")
    if rules.empty:
        st.warning("Tidak ada aturan. Turunkan minimum support/confidence."); return
    st.markdown("**f. Maksimal 10 aturan terbaik (urut Lift, lalu Confidence)**")
    top = rules.head(10); show_df(fmt_rules(top))
    st.markdown("**g. Visualisasi**")
    c = st.columns(2)
    with c[0]: network(top)
    with c[1]:
        pv = top.pivot_table(index="Antecedent", columns="Consequent", values="Lift")
        show_fig(px.imshow(pv, text_auto=".2f", color_continuous_scale="Reds", title="Heatmap Lift (antecedent × consequent)"), 480)

    soal("Soal 6. Interpretasi Pola Pembelian")
    t1 = top.iloc[0]; t3 = top.head(3)
    pairs = list(dict.fromkeys(" & ".join(sorted([r.Antecedent, r.Consequent])) for r in top.itertuples()))[:5]
    jawaban("s6a", "a. Kombinasi produk paling menarik untuk bisnis.", "; ".join(pairs) + f". Pasangan ini memiliki lift tertinggi (≈{f2(t1['Lift'])}).", 100)
    jawaban("s6b", "b. Arti support, confidence, dan lift pada aturan terpilih.",
            f"Aturan {t1['Antecedent']} → {t1['Consequent']}: Support {pct(t1['Support'], 2)} = pasangan muncul pada {pct(t1['Support'], 2)} dari semua transaksi. "
            f"Confidence {pct(t1['Confidence'])} = dari transaksi yang memuat {t1['Antecedent']}, {pct(t1['Confidence'])} juga memuat {t1['Consequent']}. "
            f"Lift {f2(t1['Lift'])} = peluang dibeli bersama {f2(t1['Lift'])}× lebih tinggi dibanding jika dibeli independen (lift > 1 = asosiasi positif).", 150)
    jawaban("s6c", "c. Minimal tiga rekomendasi cross-selling / bundling.",
            "\n".join(f"{i + 1}) Paket {r.Antecedent} + {r.Consequent}: tawarkan {r.Consequent} saat {r.Antecedent} dibeli (confidence {pct(r.Confidence)}, lift {f2(r.Lift)})."
                      for i, r in enumerate(t3.itertuples())), 130)
    jawaban("s6d", "d. Produk yang layak ditempatkan berdekatan.", "Tempatkan berdekatan: " + "; ".join(pairs) + ".", 100)
    jawaban("s6e", "e. Alasan setiap rekomendasi.",
            "Setiap rekomendasi bersumber dari aturan dengan lift > 3 dan support ≥ 1% sehingga bukan perkiraan: pembelian bersama terjadi jauh lebih sering daripada "
            "kebetulan dan melibatkan ratusan transaksi nyata.", 100)


# ------------------------------------------------------------------ TAHAP 4
def page_t4(R):
    ek, rfm = R["elbow"], R["rfm"]
    st.header("Tahap 4 — Segmentasi Pelanggan Menggunakan K-Means (20%)")
    soal("Soal 7. Pemodelan Segmentasi Pelanggan")
    st.markdown("**a. Fitur**")
    st.markdown(f"Recency, Frequency, Monetary (log + standardisasi) untuk **{num(len(rfm))}** pelanggan.")
    st.markdown("**b & c. Elbow Method dan Silhouette Score**")
    show_df(ek.round(4))
    c = st.columns(2)
    with c[0]:
        f = px.line(ek, x="k", y="Inertia", markers=True, title="Grafik Elbow"); f.update_traces(line=dict(color=ACCENT, width=4)); show_fig(f, 340)
    with c[1]:
        f = px.line(ek, x="k", y="Silhouette", markers=True, title="Silhouette Score per k"); f.update_traces(line=dict(color="#2A9D8F", width=4)); show_fig(f, 340)
    kbest = int(ek.loc[ek["Silhouette"].idxmax(), "k"])
    st.markdown("**d. Jumlah klaster**")
    k = st.selectbox("Pilih k", list(range(2, 9)), index=st.session_state.get("k_final", max(3, kbest)) - 2,
                     help="Default k = 3: kompromi antara Silhouette dan kebutuhan tiga strategi bisnis (Loyal, Potential, Churn Risk).")
    st.session_state["k_final"] = k
    sil = ek.set_index("k")["Silhouette"]
    st.success(f"**k = {k}** (Silhouette {sil[k]:.3f}; Silhouette tertinggi pada k = {kbest}). Siku kurva inertia terlihat di sekitar k = 3; "
               "k lebih besar menurunkan kualitas pemisahan dan menyulitkan strategi pemasaran.")
    seg, prof, it, ine = seg_data(R)
    st.markdown("**e. Hasil pengelompokan**")
    kpi([("k", k), ("Inertia", f"{ine:.1f}"), ("Iterasi", it), ("Silhouette", f"{sil[k]:.3f}")])
    show_df(prof[["Segmen", "Pelanggan", "% Pelanggan"]].round(1))
    st.markdown("**f. Visualisasi klaster**")
    c = st.columns(2)
    with c[0]: show_fig(px.scatter(seg, x="Recency", y="Monetary", color="Segmen", color_discrete_map=SEG_COLOR, hover_data=["Customer_ID", "Frequency"],
                                   log_y=True, title="Recency vs Monetary"), 420)
    with c[1]: show_fig(px.scatter(seg, x="Frequency", y="Monetary", color="Segmen", color_discrete_map=SEG_COLOR, hover_data=["Customer_ID"],
                                   log_y=True, title="Frequency vs Monetary"), 420)
    show_fig(px.scatter_3d(seg, x="Z_Recency", y="Z_Frequency", z="Z_Monetary", color="Segmen", color_discrete_map=SEG_COLOR, title="Klaster pada ruang standar (3D)"), 500)

    soal("Soal 8. Interpretasi dan Strategi Pelanggan")
    st.markdown("**a–c. Profil, label, dan karakteristik klaster**")
    pt = prof.copy(); pt["Karakteristik"] = pt["Segmen"].map(lambda s: KARAKTER[kind(s)])
    for col in ["Recency", "Frequency"]: pt[col] = pt[col].round(2)
    pt["Monetary"] = pt["Monetary"].map(rp)
    show_df(pt[["Segmen", "Pelanggan", "Recency", "Frequency", "Monetary", "Karakteristik"]])
    st.markdown("**d. Strategi pemasaran per segmen**")
    for r in prof.itertuples():
        st.markdown(f'<div class="seg-card" style="border-top-color:{SEG_COLOR.get(r.Segmen, ACCENT)}"><div class="t">{r.Segmen} • {int(r.Pelanggan)} pelanggan '
                    f'({r._7:.1f}%) • {r._8:.1f}% revenue</div><div class="d">{STRATEGI[kind(r.Segmen)]}</div></div>', unsafe_allow_html=True)
    pri = prof.sort_values("Pelanggan", ascending=False).iloc[0]
    jawaban("s8e", "e. Segmen prioritas dan alasannya.",
            f"Prioritas: {pri['Segmen']} ({int(pri['Pelanggan'])} pelanggan, {pri['% Revenue']:.1f}% revenue). Segmen ini terbesar dan paling berpengaruh pada pendapatan sehingga "
            "perubahan perilakunya memberi dampak terbesar. Berikutnya Loyal Customers untuk menjaga kontribusi, lalu Churn Risk bila anggaran tersedia.", 130)


# ------------------------------------------------------------------ TAHAP 5
def page_t5(R):
    seg, prof, _, _ = seg_data(R)
    d = R["clean"].merge(seg[["Customer_ID", "Segmen"]], on="Customer_ID", how="left").fillna({"Segmen": "Tidak Teridentifikasi"})
    st.header("Tahap 5 — Business Intelligence Dashboard (15%)")
    soal("Soal 9. Dashboard Interaktif")
    c = st.columns(3)
    b0, b1 = c[0].select_slider("Filter periode transaksi", BULAN, value=(BULAN[0], BULAN[-1]))
    cat = c[1].multiselect("Filter kategori", sorted(d["Category"].unique()), default=sorted(d["Category"].unique()))
    sg = c[2].multiselect("Filter segmen pelanggan", sorted(d["Segmen"].unique()), default=sorted(d["Segmen"].unique()))
    i0, i1 = BULAN.index(b0) + 1, BULAN.index(b1) + 1
    f = d[d["Bulan"].between(i0, i1) & d["Category"].isin(cat) & d["Segmen"].isin(sg)]
    if f.empty:
        st.warning("Tidak ada data untuk filter ini."); return
    bs = f.groupby("Transaction_ID")["Product_ID"].nunique().mean()
    st.subheader("Key Performance Indicators")
    kpi([("Total pendapatan", rp(f["Line_Total"].sum())), ("Jumlah transaksi", num(f["Transaction_ID"].nunique())),
         ("Pelanggan unik aktif", num(f["Customer_ID"].nunique())), ("Rata-rata basket_size", f2(bs))])
    st.subheader("Analisis Penjualan")
    m = f.groupby("Bulan", as_index=False)["Line_Total"].sum(); m["Nama"] = m["Bulan"].map(lambda i: BULAN[i - 1])
    c = st.columns(2)
    with c[0]: show_fig(px.line(m, x="Nama", y="Line_Total", markers=True, title="Tren penjualan bulanan").update_xaxes(categoryorder="array", categoryarray=BULAN), 340)
    top = f.groupby("Product_Name")["Line_Total"].sum().nlargest(10).reset_index()
    with c[1]: show_fig(px.bar(top, x="Line_Total", y="Product_Name", orientation="h", title="10 produk penjualan tertinggi").update_yaxes(autorange="reversed"), 340)
    c = st.columns(2)
    with c[0]: show_fig(px.pie(f.groupby("Category", as_index=False)["Line_Total"].sum(), names="Category", values="Line_Total", hole=.45, title="Distribusi penjualan per kategori"), 340)
    with c[1]: show_fig(px.bar(f.groupby("Region", as_index=False)["Line_Total"].sum(), x="Region", y="Line_Total", title="Penjualan per wilayah"), 340)
    st.subheader("Analisis Apriori")
    key = (R["fid"], i0, i1, tuple(cat), tuple(sg))
    ms = st.slider("Minimum support (dashboard)", 0.002, 0.05, 0.01, 0.002, format="%.3f", key="dms")
    n, _, rules = cached_apriori(f, key, ms, 0.2)
    rules = rules[rules["Lift"] > 1].head(10)
    if rules.empty: st.info("Belum ada aturan pada filter ini; turunkan minimum support.")
    else:
        c = st.columns(2)
        with c[0]: show_df(fmt_rules(rules))
        with c[1]: network(rules, 400)
    st.subheader("Analisis K-Means")
    sf = seg[seg["Customer_ID"].isin(f["Customer_ID"].unique())]
    c = st.columns(2)
    with c[0]: show_fig(px.bar(sf["Segmen"].value_counts().reset_index(), x="Segmen", y="count", color="Segmen", color_discrete_map=SEG_COLOR, title="Pelanggan per klaster"), 340)
    with c[1]: show_fig(px.scatter(sf, x="Recency", y="Monetary", color="Segmen", color_discrete_map=SEG_COLOR, log_y=True, title="Visualisasi segmentasi"), 340)
    pt = prof.copy(); pt["Strategi"] = pt["Segmen"].map(lambda s: STRATEGI[kind(s)])
    show_df(pt[["Segmen", "Pelanggan", "Recency", "Frequency", "Monetary", "Strategi"]].round(1))
    st.subheader("Bonus — Integrasi Apriori × K-Means")
    bn = bonus_rules(R, seg)
    bf = fmt_rules(bn.drop(columns="Segmen")).drop(columns="No")
    bf.insert(0, "Segmen", bn["Segmen"].values)
    show_df(bf)
    note("<b>Cara membaca:</b> KPI menunjukkan kondisi bisnis pada periode terpilih; tren dan top produk menunjukkan kapan dan apa yang laku; aturan Apriori menunjukkan "
         "kombinasi produk untuk bundling; klaster menunjukkan siapa pelanggan bernilai dan siapa yang berisiko pergi.")


# ------------------------------------------------------------------ TAHAP 6
def page_t6(R):
    d = R["clean"]; seg, prof, _, _ = seg_data(R)
    _, _, rules = cached_apriori(d, (R["fid"], "all"), 0.01, 0.2)
    rules = rules[rules["Lift"] > 1]
    st.header("Tahap 6 — Rekomendasi Keputusan Bisnis (10%)")
    soal("Soal 10. Penyusunan Rekomendasi Preskriptif")
    if rules.empty: st.warning("Aturan Apriori tidak ditemukan."); return
    t3 = rules.head(3); n = d["Transaction_ID"].nunique(); sales = d["Line_Total"].sum()
    loy, chu = prof.iloc[0], prof.iloc[-1]; mid = prof.iloc[1] if len(prof) > 2 else prof.iloc[0]
    st.markdown("**a. Strategi penempatan produk**")
    for r in t3.itertuples(): st.markdown(f"- Tempatkan **{r.Antecedent}** berdekatan dengan **{r.Consequent}** dan tawarkan bundling (lift {f2(r.Lift)}).")
    st.markdown("**b. Strategi retensi pelanggan**")
    st.markdown(f"- **{loy['Segmen']}** ({int(loy['Pelanggan'])}): {STRATEGI[kind(loy['Segmen'])]}\n- **{chu['Segmen']}** ({int(chu['Pelanggan'])}): {STRATEGI[kind(chu['Segmen'])]}")
    st.markdown("**c. Prioritas tindakan**")
    show_df(pd.DataFrame([[1, f"Bundling & cross-sell {t3.iloc[0]['Antecedent']} + {t3.iloc[0]['Consequent']}", "Tinggi", "Tinggi", "Rendah", "Mudah"],
                          [2, f"Reaktivasi segmen {chu['Segmen']}", "Tinggi", "Sedang", "Sedang", "Sedang"],
                          [3, f"Program loyalitas {loy['Segmen']}", "Sedang", "Tinggi", "Sedang", "Sedang"],
                          [4, "Penataan rak berdasarkan aturan Apriori", "Sedang", "Sedang", "Rendah", "Mudah"]],
                         columns=["#", "Rekomendasi", "Urgensi", "Potensi manfaat", "Biaya", "Kemudahan"]))
    st.markdown("**d. Estimasi dampak bisnis (berdasarkan ASUMSI)**")
    c = st.columns(3)
    conv = c[0].slider("Asumsi konversi bundling", 0.01, 0.20, 0.05, 0.01)
    rea = c[1].slider("Asumsi reaktivasi Churn Risk", 0.02, 0.30, 0.10, 0.02)
    upg = c[2].slider("Asumsi upgrade Potential → Loyal", 0.02, 0.30, 0.10, 0.02)
    rows, tot = [], 0
    for r in t3.itertuples():
        ant = r.Support / r.Confidence * n; gap = ant * (1 - r.Confidence)
        dd = d[d["Product_Name"] == r.Consequent.split(" + ")[0]]; ap_ = dd["Line_Total"].sum() / max(dd["Quantity"].sum(), 1)
        est = gap * conv * ap_; tot += est
        rows.append([f"Bundling {r.Antecedent} → {r.Consequent}", f"≈{num(ant)} transaksi memuat antecedent; {num(gap)} belum memuat consequent", f"{pct(conv, 0)} menambah consequent", rp(est)])
    e2 = chu["Pelanggan"] * rea * (mid["Monetary"] - chu["Monetary"]); e3 = mid["Pelanggan"] * upg * (loy["Monetary"] - mid["Monetary"]); tot += e2 + e3
    rows += [[f"Reaktivasi {chu['Segmen']}", f"{int(chu['Pelanggan'])} pelanggan", f"{pct(rea, 0)} kembali ke level {mid['Segmen']}", rp(e2)],
             [f"Upgrade {mid['Segmen']}", f"{int(mid['Pelanggan'])} pelanggan", f"{pct(upg, 0)} naik ke level {loy['Segmen']}", rp(e3)],
             ["TOTAL", "", "", f"{rp(tot)} ({pct(tot / sales)} dari penjualan 2025)"]]
    show_df(pd.DataFrame(rows, columns=["Rekomendasi", "Dasar dari data", "Asumsi", "Estimasi tambahan pendapatan / tahun"]))
    note("<b>Bedakan:</b> (1) hasil dari data = support, confidence, lift, profil RFM; (2) estimasi = angka di atas berdasarkan asumsi konversi; "
         "(3) dampak aktual baru diketahui setelah implementasi dan uji (A/B test).")
    st.markdown("**e. Rencana implementasi**")
    show_df(pd.DataFrame([["Jangka pendek (0–3 bulan)", f"Bundling & penataan rak untuk 3 aturan teratas; kampanye WhatsApp reaktivasi {chu['Segmen']}; validasi input kasir (Quantity & harga)."],
                          ["Jangka menengah (3–12 bulan)", "Program poin/loyalitas, perbarui segmentasi RFM tiap kuartal, hubungkan dashboard ke data transaksi harian."]],
                         columns=["Jangka waktu", "Tindakan"]))
    st.markdown("**f. KPI evaluasi**")
    st.markdown("- Attach rate pasangan produk bundling.\n- Rata-rata basket_size dan pendapatan per transaksi.\n- % pelanggan Churn Risk yang kembali bertransaksi dalam 60 hari.\n"
                "- Frekuensi belanja dan pertumbuhan pendapatan per segmen.\n- Pertumbuhan penjualan bulanan terhadap 2025.")


# ------------------------------------------------------------------ LUARAN
def page_luaran(R):
    d = R["clean"]; seg, prof, it, ine = seg_data(R)
    st.header("Luaran & Export")
    _, isets, rules = cached_apriori(d, (R["fid"], "all"), 0.01, 0.2)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        d.drop(columns="Bulan").to_excel(w, sheet_name="Data_Bersih", index=False)
        R["log"].to_excel(w, sheet_name="Dokumentasi_Cleaning", index=False)
        R["ba"].to_excel(w, sheet_name="Sebelum_vs_Sesudah", index=False)
        seg.to_excel(w, sheet_name="RFM_Segmen", index=False)
        rules.to_excel(w, sheet_name="Aturan_Apriori", index=False)
        prof.to_excel(w, sheet_name="Profil_Klaster", index=False)
    st.download_button("⬇️ Download Excel hasil (dataset preprocessing + dokumentasi + hasil model)", buf.getvalue(), "UTS_BI_Hasil.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c = st.columns(3)
    c[0].download_button("⬇️ Data bersih (CSV)", d.to_csv(index=False).encode(), "data_bersih.csv", "text/csv")
    c[1].download_button("⬇️ RFM + segmen (CSV)", seg.to_csv(index=False).encode(), "rfm_segmen.csv", "text/csv")
    c[2].download_button("⬇️ Aturan Apriori (CSV)", rules.to_csv(index=False).encode(), "aturan_apriori.csv", "text/csv")
    st.subheader("Checklist luaran")
    ck = ["Laporan analisis (Tahap 1–6)", "Dataset hasil preprocessing + dokumentasi", "Kode program (file ini)", "Dashboard BI (Tahap 5)",
          "Dokumentasi visualisasi", "Presentasi hasil"]
    show_df(pd.DataFrame({"No": range(1, 7), "Luaran": ck, "Status": ["Di aplikasi", "☑ Export Excel", "☑ Selesai", "☑ Selesai", "☑ Di aplikasi", "Siapkan dari hasil aplikasi"]}))
    st.subheader("Kesimpulan")
    st.markdown(f"- Data mentah {num(len(R['raw']))} baris dibersihkan menjadi {num(len(d))} baris valid dari {num(d['Customer_ID'].nunique())} pelanggan.")
    if not rules.empty: st.markdown(f"- Apriori menemukan {len(rules[rules['Lift'] > 1])} aturan (lift > 1); terkuat: {rules.iloc[0]['Antecedent']} → {rules.iloc[0]['Consequent']} (lift {f2(rules.iloc[0]['Lift'])}).")
    for r in prof.itertuples(): st.markdown(f"- {r.Segmen}: {int(r.Pelanggan)} pelanggan, {r._8:.1f}% revenue.")


PAGES = [page_ringkasan, page_t1, page_t2, page_t3, page_t4, page_t5, page_t6, page_luaran]


# ------------------------------------------------------------------ COVER / LOGIN / MAIN
def read_first(files):
    for f in files:
        if os.path.exists(f):
            with open(f, "rb") as fh: return fh.read()
    return None


def foto_bytes(): return read_first(FOTO_FILE) or base64.b64decode(FOTO_PROFIL)


def img_uri(raw): return ("data:image/png;base64," if raw[:4] == b"\x89PNG" else "data:image/jpeg;base64,") + base64.b64encode(raw).decode()
def hide_sidebar(): st.markdown("<style>[data-testid='stSidebar'],[data-testid='stSidebarCollapsedControl'],[data-testid='collapsedControl']{display:none;}</style>", unsafe_allow_html=True)


def page_cover():
    hide_sidebar()
    logo, foto = read_first(LOGO_FILE), foto_bytes()
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="cover"><div class="cover-logo">' + (f'<img src="{img_uri(logo)}">' if logo else "🎓") + "</div>"
                    f'<div class="cover-univ">{html.escape(UNIVERSITAS)}</div><div class="cover-fak">{html.escape(FAKULTAS)} &bull; {html.escape(PRODI)}</div>'
                    '<div class="cover-line"></div>'
                    f'<div class="cover-mk">Ujian Tengah Semester - {html.escape(MATA_KULIAH)}</div>'
                    '<div class="cover-title">INTEGRASI DATA MINING &amp; BI<br>PADA DATA TRANSAKSI TOKO</div>'
                    '<div class="cover-sub">EDA &bull; Preprocessing &bull; Apriori &bull; K-Means &bull; Dashboard BI</div>'
                    + (f'<img class="cover-foto" src="{img_uri(foto)}">' if foto else '<div class="cover-foto">👤</div>') +
                    f'<div class="cover-nama">{html.escape(NAMA)}</div><div class="cover-npm">NPM {html.escape(NPM)}</div></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Masuk ke Dashboard", type="primary", use_container_width=True):
            st.session_state["stage"] = "app"; st.rerun()


def main():
    st.set_page_config(page_title="UTS Business Intelligence", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    if st.session_state.get("stage", "cover") == "cover": return page_cover()
    with st.sidebar:
        st.markdown(f'<img src="{img_uri(foto_bytes())}" style="width:96px;height:96px;border-radius:50%;object-fit:cover;border:3px solid #E63946;">', unsafe_allow_html=True)
        st.markdown(f"**{NAMA}**  \nNPM {NPM}")
        st.markdown("## 📊 UTS Business Intelligence")
        st.caption("Data Transaksi Toko")
        up = st.file_uploader("Upload file Excel", type=["xlsx"], help=f"Jika tidak di-upload, aplikasi mencari file: {EXCEL_DEFAULT}")
        st.markdown("---"); st.markdown("**MENU**")
        menu = st.radio("Menu", MENU, label_visibility="collapsed")
        st.markdown("---")
        if st.button("← Kembali ke Cover", use_container_width=True):
            st.session_state["stage"] = "cover"; st.rerun()
    st.title("INTEGRASI DATA MINING & BUSINESS INTELLIGENCE — TOKO")
    if up is not None: data, src = up.getvalue(), up.name
    elif os.path.exists(EXCEL_DEFAULT):
        data, src = open(EXCEL_DEFAULT, "rb").read(), EXCEL_DEFAULT
    else:
        note(f"⬅️ Upload file Excel di sidebar, atau letakkan <b>{html.escape(EXCEL_DEFAULT)}</b> di folder yang sama dengan app."); return
    st.caption(f"Sumber data: {src}")
    try: R = cached_etl(data)
    except Exception as e: st.exception(e); return
    PAGES[MENU.index(menu)](R)


if __name__ == "__main__":
    main()
