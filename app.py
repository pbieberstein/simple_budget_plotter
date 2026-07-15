import streamlit as st
import pandas as pd
import io
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Spend Split Dashboard",
    page_icon="$",
    layout="wide",
)


def _titlecase_person(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    # Keep the header-provided name as-is (trimmed)
    return name.strip()


def load_new_budget_export(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse a budget CSV where columns follow the pattern:
    - Paid by <NameA>, Paid by <NameB>
    - Paid for <NameA>, Paid for <NameB>
    - Date, Title, Category, Currency, Exchange Rate (names may vary in case)

    Dynamically extracts the two names from column headers and returns a normalized
    long DataFrame with columns: date, title, category, person, amount.

    Amounts are converted by dividing by the Exchange Rate (if present). If no
    named exchange-rate column exists, the last column is used as FX when numeric.
    """
    cols_list = list(df_raw.columns)
    # Locate key columns by case-insensitive matching
    def find_col(name_ci: str):
        for c in cols_list:
            if c.strip().lower() == name_ci:
                return c
        return None

    col_date = find_col("date")
    col_title = find_col("title")
    col_cat = find_col("category")
    col_curr = find_col("currency")
    col_fx = find_col("exchange rate")

    if not (col_date and col_title and col_cat):
        raise ValueError("Not the new budget export schema: missing Date/Title/Category")

    # Identify dynamic person columns
    paid_for_cols = []
    paid_by_cols = []
    for c in cols_list:
        cl = c.strip().lower()
        if cl.startswith("paid for "):
            paid_for_cols.append(c)
        elif cl.startswith("paid by "):
            paid_by_cols.append(c)

    if len(paid_for_cols) == 0 and len(paid_by_cols) == 0:
        raise ValueError("Not the new budget export schema: no 'Paid for'/'Paid by' columns")

    # Determine person order from header order
    person_order = []
    if paid_for_cols:
        for c in cols_list:
            low = c.strip().lower()
            if low.startswith("paid for "):
                person_order.append(c[len("Paid for "):].strip())
    elif paid_by_cols:
        for c in cols_list:
            low = c.strip().lower()
            if low.startswith("paid by "):
                person_order.append(c[len("Paid by "):].strip())
    # Keep only first two distinct names (expected two)
    seen = set()
    person_order = [_titlecase_person(p) for p in person_order if not (p in seen or seen.add(p))][:2]

    # Parse date
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df[col_date].astype(str), format="%Y%m%d", errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df[col_date], errors="coerce")

    # Determine exchange rate series
    if col_fx and col_fx in df.columns:
        fx_series = pd.to_numeric(df[col_fx], errors="coerce")
    else:
        # Fallback to last column if numeric
        last_col = df.columns[-1]
        fx_series = pd.to_numeric(df[last_col], errors="coerce") if last_col not in {col_date, col_title, col_cat} else pd.Series([1.0] * len(df))
    fx_series = fx_series.fillna(1.0).replace(0, 1.0)  # avoid div by zero

    # Build long rows from 'Paid for' columns; if absent, we can fallback to 'Paid by'
    long_rows = []
    def append_rows_from_cols(cols, prefix_lower):
        for c in cols:
            # Extract display name as text after the prefix
            cl = c.strip()
            low = cl.lower()
            if not low.startswith(prefix_lower):
                continue
            person_name = cl[len(prefix_lower):].strip()
            for idx, row in df.iterrows():
                date = row.get("date")
                title = row.get(col_title)
                category = row.get(col_cat)
                currency = row.get(col_curr) if col_curr in df.columns else None
                rate = fx_series.iloc[idx] if isinstance(fx_series, pd.Series) else 1.0
                amt_raw = pd.to_numeric(row.get(c, 0), errors="coerce")
                if pd.isna(amt_raw) or amt_raw == 0:
                    continue
                amount = float(amt_raw) / float(rate)
                long_rows.append({
                    "date": date,
                    "title": title,
                    "category": category,
                    "person": _titlecase_person(person_name),
                    "amount": amount,
                    "currency": currency,
                })

    if len(paid_for_cols) > 0:
        append_rows_from_cols(paid_for_cols, "paid for ")
    else:
        # Fallback: use 'Paid by' columns as a proxy if 'Paid for' missing
        append_rows_from_cols(paid_by_cols, "paid by ")

    long_df = pd.DataFrame(long_rows)
    long_df = long_df[pd.notna(long_df["date"])].copy()
    # Persist detected order for downstream plots
    if person_order:
        st.session_state["person_order"] = person_order
    return long_df


def load_legacy_expenses(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Parse the legacy CSV found in this repo with columns:
    Title, Category, amount, Paid By, How to Split, date

    Returns long DataFrame with: date, title, category, person, amount
    """
    expected = {"Title", "Category", "amount", "Paid By", "How to Split", "date"}
    if not expected.issubset(set(df_raw.columns)):
        raise ValueError("Not the legacy schema")

    df = df_raw.copy()
    # Parse date like "12/19/2024, 12:00:00 AM"
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y, %I:%M:%S %p", errors="coerce")
    if df["date"].isna().any():
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    # Deduce persons from Paid By column (expect up to 2)
    seen = []
    for p in df["Paid By"].dropna().tolist():
        if p not in seen:
            seen.append(p)
    if len(seen) == 0:
        persons = ["Person 1", "Person 2"]
    elif len(seen) == 1:
        persons = [seen[0], "Person 2"]
    else:
        persons = seen[:2]
    # Persist order
    st.session_state["person_order"] = [_titlecase_person(p) for p in persons]

    def split_row(row):
        amt = row["amount"]
        paid_by = row["Paid By"]
        split = str(row["How to Split"]).strip()
        p1, p2 = persons[0], persons[1]
        if split == "Split Evenly":
            return [(p1, amt/2 if paid_by == p1 else amt/2), (p2, amt/2 if paid_by == p2 else amt/2)]
        elif split == "The wrong person paid":
            # The full amount should count toward the other person
            other = p2 if paid_by == p1 else p1
            return [(other, amt)]
        elif split == "The correct person paid":
            return [(paid_by, amt)]
        else:
            # Fallback: attribute to payer
            return [(paid_by, amt)]

    long_rows = []
    for _, row in df.iterrows():
        for person, amount in split_row(row):
            long_rows.append({
                "date": row["date"],
                "title": row["Title"],
                "category": row["Category"],
                "person": _titlecase_person(person),
                "amount": float(amount),
                "currency": None,
            })

    long_df = pd.DataFrame(long_rows)
    long_df = long_df[pd.notna(long_df["date"])]
    return long_df


def _read_csv_robust(uploaded_file) -> pd.DataFrame:
    """Read CSVs from various OS/Excel exports robustly.
    - Detect BOM for UTF-16/UTF-8-SIG
    - Try common encodings if needed
    - Auto-detect separator
    """
    data = uploaded_file.read()
    # Heuristic encoding detection via BOM
    enc_guess = None
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        enc_guess = "utf-16"
    elif data.startswith(b"\xef\xbb\xbf"):
        enc_guess = "utf-8-sig"
    tried = []
    for enc in [enc_guess, "utf-8", "utf-8-sig", "utf-16", "utf-16le", "utf-16be", "latin-1"]:
        if not enc:
            continue
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
            return df
        except Exception as e:
            tried.append((enc, str(e)))
            continue
    # Last resort: try default without encoding
    df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    return df


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df_raw = _read_csv_robust(uploaded_file)
    # Try new schema first
    try:
        return load_new_budget_export(df_raw)
    except Exception:
        pass
    # Fallback to legacy schema
    try:
        return load_legacy_expenses(df_raw)
    except Exception as e:
        raise ValueError(f"Unsupported CSV format: {e}")


def clean_budget_data(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["person"] = df["person"].map(_titlecase_person)
    df["category"] = df["category"].fillna("Uncategorized").astype(str).str.strip()
    df.loc[df["category"].eq("") | df["category"].str.lower().eq("nan"), "category"] = "Uncategorized"
    df["title"] = df["title"].fillna("").astype(str)
    return df[pd.notna(df["date"]) & df["amount"].ne(0)].copy()


def get_person_order(df: pd.DataFrame) -> list[str]:
    detected = [_titlecase_person(p) for p in st.session_state.get("person_order", [])]
    present = df["person"].dropna().map(_titlecase_person).unique().tolist()
    ordered = [p for p in detected if p in present]
    ordered.extend([p for p in sorted(present) if p not in ordered])
    return ordered


def make_color_maps(person_order: list[str], categories: list[str]):
    person_colors = ["#e15759", "#4e79a7", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2"]
    category_colors = (
        px.colors.qualitative.Bold
        + px.colors.qualitative.Set2
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Safe
    )
    person_map = {person: person_colors[i % len(person_colors)] for i, person in enumerate(person_order)}
    category_map = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(categories)}
    return person_map, category_map


def money(value: float) -> str:
    return f"${value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.1f}%"


def style_currency(styler, cols: list[str]):
    return styler.format({col: "${:,.2f}" for col in cols if col in styler.data.columns})


def apply_plot_style(fig: go.Figure, height: int = 430) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=54, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9dee8")
    fig.update_yaxes(gridcolor="#eef1f6", zerolinecolor="#cfd6e3")
    return fig


def apply_app_css():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }
            [data-testid="stMetric"] {
                background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
                border: 1px solid #e4e9f2;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }
            [data-testid="stMetricLabel"] {
                color: #526070;
                font-size: 0.82rem;
            }
            [data-testid="stMetricValue"] {
                color: #172033;
                font-weight: 750;
            }
            div[data-testid="stTabs"] button p {
                font-weight: 650;
            }
            .stDataFrame {
                border: 1px solid #e6ebf3;
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def complete_monthly_person_grid(monthly: pd.DataFrame, months: pd.DatetimeIndex, persons: list[str]) -> pd.DataFrame:
    if not persons:
        return monthly
    grid = pd.MultiIndex.from_product([months, persons], names=["Month", "Person"])
    return (
        monthly.set_index(["Month", "Person"])
        .reindex(grid, fill_value=0.0)
        .reset_index()
    )


def category_difference_table(lf: pd.DataFrame, person_order: list[str]) -> pd.DataFrame:
    pivot = (
        lf.pivot_table(index="category", columns="person", values="amount", aggfunc="sum", fill_value=0.0)
        .reindex(columns=person_order, fill_value=0.0)
        .reset_index()
        .rename(columns={"category": "Category"})
    )
    if not person_order:
        return pivot

    p1 = person_order[0]
    p2 = person_order[1] if len(person_order) > 1 else None
    pivot["Total"] = pivot[person_order].sum(axis=1)
    if p2:
        pivot["Difference"] = pivot[p1] - pivot[p2]
        pivot["Abs Difference"] = pivot["Difference"].abs()
        pivot["Higher spender"] = pivot["Difference"].apply(lambda x: p1 if x > 0 else (p2 if x < 0 else "Tie"))
        pivot[f"{p1} share"] = pivot[p1] / pivot["Total"].replace(0, pd.NA)
        pivot[f"{p2} share"] = pivot[p2] / pivot["Total"].replace(0, pd.NA)
    else:
        pivot["Difference"] = pivot[p1]
        pivot["Abs Difference"] = pivot[p1].abs()
        pivot["Higher spender"] = p1
        pivot[f"{p1} share"] = 1.0
    return pivot.sort_values("Abs Difference", ascending=False)


def year_month_range(months: list[pd.Period], year: int):
    year_months = [month for month in months if month.year == year]
    if not year_months:
        return None
    return str(year_months[0]), str(year_months[-1])


def trailing_month_range(months: list[pd.Period], count: int):
    if not months:
        return None
    selected = months[-count:]
    return str(selected[0]), str(selected[-1])


def default_selected_categories(categories: list[str]) -> list[str]:
    selected = [category for category in categories if category.strip().lower() != "money transfer"]
    return selected if selected else categories


def month_name(month: int) -> str:
    return pd.Timestamp(year=2000, month=int(month), day=1).strftime("%b")


def month_count(month_range: tuple[int, int]) -> int:
    return max(1, int(month_range[1]) - int(month_range[0]) + 1)


def in_month_range(df: pd.DataFrame, month_range: tuple[int, int]) -> pd.Series:
    months = df["date"].dt.month
    return (months >= int(month_range[0])) & (months <= int(month_range[1]))


def reset_invalid_month_range(key: str, default_value: tuple[int, int]):
    value = st.session_state.get(key)
    try:
        start_month, end_month = int(value[0]), int(value[1])
        is_valid = 1 <= start_month <= end_month <= 12
    except (TypeError, ValueError, IndexError):
        is_valid = False
    if is_valid:
        st.session_state[key] = (start_month, end_month)
    else:
        st.session_state[key] = default_value


def render_range_button(label: str, target_range, key: str):
    if st.button(label, key=key, disabled=target_range is None, use_container_width=True):
        st.session_state["month_range"] = target_range


def render_empty_state():
    st.info("Upload a budget CSV from the sidebar to build the dashboard.")


apply_app_css()

st.title("Spend Split Dashboard")
st.caption("Compare two people across time, categories, and transaction detail.")

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    render_empty_state()
else:
    try:
        long_df = clean_budget_data(load_uploaded_csv(uploaded_file))
    except Exception as e:
        st.error(str(e))
        st.stop()

    if long_df.empty:
        st.warning("No non-zero expenses were found in the uploaded CSV.")
        st.stop()

    person_order = get_person_order(long_df)
    categories = sorted(long_df["category"].dropna().unique().tolist())
    person_color_map, category_color_map = make_color_maps(person_order, categories)

    months_all = sorted(pd.to_datetime(long_df["date"]).dt.to_period("M").unique())
    month_labels = [str(m) for m in months_all]
    default_range = (month_labels[0], month_labels[-1])

    with st.sidebar:
        st.header("Filters")
        if "month_range" not in st.session_state or any(m not in month_labels for m in st.session_state["month_range"]):
            st.session_state["month_range"] = default_range

        current_year = pd.Timestamp.today().year
        st.caption("Quick ranges")
        quick_left, quick_right = st.columns(2)
        with quick_left:
            render_range_button("All", default_range, "range_all")
            render_range_button(f"Current year ({current_year})", year_month_range(months_all, current_year), "range_current_year")
            render_range_button("Year 2024", year_month_range(months_all, 2024), "range_2024")
        with quick_right:
            render_range_button("Last 3 months", trailing_month_range(months_all, 3), "range_last_3")
            render_range_button("Last 6 months", trailing_month_range(months_all, 6), "range_last_6")
            render_range_button(f"Last year ({current_year - 1})", year_month_range(months_all, current_year - 1), "range_last_year")

        start_label, end_label = st.select_slider(
            "Month range",
            options=month_labels,
            value=st.session_state["month_range"],
            key="month_range",
        )
        selected_categories = st.multiselect(
            "Categories",
            categories,
            default=default_selected_categories(categories),
        )
        show_cumulative = st.toggle("Cumulative race", value=True)
        max_top = min(15, len(categories))
        if max_top > 1:
            top_n = st.slider("Top categories", 1, max_top, min(8, max_top))
        else:
            top_n = 1

    start_period = pd.Period(start_label, freq="M")
    end_period = pd.Period(end_label, freq="M")
    date_start = start_period.start_time
    date_end = end_period.end_time

    month_series = long_df["date"].dt.to_period("M")
    lf = long_df[
        (month_series >= start_period)
        & (month_series <= end_period)
        & (long_df["category"].isin(selected_categories))
    ].copy()

    if lf.empty:
        st.warning("No expenses match the current filters.")
        st.stop()

    months_window = pd.period_range(start=start_period, end=end_period, freq="M").to_timestamp()
    totals_by_person = lf.groupby("person")["amount"].sum().reindex(person_order, fill_value=0.0)
    grand_total = totals_by_person.sum()
    txn_count = int(len(lf))
    avg_monthly = grand_total / max(1, len(months_window))

    diff_table = category_difference_table(lf, person_order)
    biggest_gap = diff_table.iloc[0] if not diff_table.empty else None
    if len(person_order) >= 2:
        person_gap = totals_by_person.iloc[0] - totals_by_person.iloc[1]
        leader = person_order[0] if person_gap > 0 else person_order[1] if person_gap < 0 else "Tie"
        leader_delta = abs(person_gap)
        leader_help = f"{pct(leader_delta / grand_total * 100) if grand_total else '0.0%'} of selected spend"
    else:
        leader = person_order[0] if person_order else "N/A"
        leader_delta = totals_by_person.iloc[0] if len(totals_by_person) else 0
        leader_help = "Only one person found"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Selected spend", money(grand_total), f"{txn_count:,} expenses")
    k2.metric("Higher total", leader, money(leader_delta) if leader != "Tie" else money(0), help=leader_help)
    k3.metric("Avg monthly spend", money(avg_monthly), f"{len(months_window)} months")
    if biggest_gap is not None:
        k4.metric("Biggest category gap", str(biggest_gap["Category"]), money(float(biggest_gap["Abs Difference"])))
    else:
        k4.metric("Biggest category gap", "N/A")

    overview_tab, yearly_tab, categories_tab, trends_tab, transactions_tab = st.tabs(
        ["Overview", "Yearly Budget", "Category Battle", "Trends", "Transactions"]
    )

    monthly_person = (
        lf.assign(Month=lf["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["Month", "person"], as_index=False)["amount"].sum()
        .rename(columns={"person": "Person", "amount": "Amount"})
    )
    monthly_person = complete_monthly_person_grid(monthly_person, months_window, person_order)
    monthly_person["Month Label"] = monthly_person["Month"].dt.strftime("%Y-%m")
    monthly_person["Cumulative"] = monthly_person.sort_values("Month").groupby("Person")["Amount"].cumsum()

    monthly_cat = (
        lf.assign(Month=lf["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["Month", "person", "category"], as_index=False)["amount"].sum()
        .rename(columns={"person": "Person", "category": "Category", "amount": "Amount"})
    )

    with overview_tab:
        left, right = st.columns([1.35, 1])
        with left:
            y_col = "Cumulative" if show_cumulative else "Amount"
            title = "Who Is Spending More Over Time" if show_cumulative else "Monthly Spend by Person"
            race = px.line(
                monthly_person,
                x="Month",
                y=y_col,
                color="Person",
                markers=True,
                color_discrete_map=person_color_map,
                category_orders={"Person": person_order},
                title=title,
                labels={y_col: "Spend"},
            )
            race.update_traces(line=dict(width=3), marker=dict(size=7))
            race.update_yaxes(tickprefix="$")
            st.plotly_chart(apply_plot_style(race, 460), use_container_width=True)

        with right:
            by_person_cat = (
                lf.groupby(["person", "category"], as_index=False)["amount"].sum()
                .rename(columns={"person": "Person", "category": "Category", "amount": "Amount"})
            )
            top_categories = (
                by_person_cat.groupby("Category", as_index=False)["Amount"].sum()
                .sort_values("Amount", ascending=False)
                .head(top_n)["Category"]
                .tolist()
            )
            donut_src = by_person_cat[by_person_cat["Category"].isin(top_categories)]
            donut = px.sunburst(
                donut_src,
                path=["Person", "Category"],
                values="Amount",
                color="Person",
                color_discrete_map=person_color_map,
                title="What Each Person Spent On",
            )
            donut.update_traces(
                textinfo="label+percent parent",
                hovertemplate="<b>%{label}</b><br>%{value:$,.2f}<extra></extra>",
            )
            st.plotly_chart(apply_plot_style(donut, 460), use_container_width=True)

        stacked = px.bar(
            monthly_cat,
            x="Month",
            y="Amount",
            color="Category",
            facet_row="Person",
            color_discrete_map=category_color_map,
            category_orders={"Person": person_order},
            title="Monthly Category Mix",
            labels={"Amount": "Spend"},
        )
        stacked.update_yaxes(tickprefix="$", matches=None)
        stacked.update_layout(legend_title_text="Category")
        st.plotly_chart(apply_plot_style(stacked, max(430, 260 * max(1, len(person_order)))), use_container_width=True)

    with yearly_tab:
        annual_source = long_df[long_df["category"].isin(selected_categories)].copy()
        if annual_source.empty:
            st.info("No yearly data is available for the selected categories.")
        else:
            annual_source["Year"] = annual_source["date"].dt.year
            annual_source["Month"] = annual_source["date"].dt.month
            available_years = sorted(annual_source["Year"].unique().tolist())
            calendar_today = pd.Timestamp.today()
            anchor_year = calendar_today.year if calendar_today.year in available_years else available_years[-1]
            prior_year_options = [year for year in available_years if year < anchor_year]
            prior_year = anchor_year - 1 if anchor_year - 1 in available_years else (prior_year_options[-1] if prior_year_options else None)

            latest_anchor_month = int(annual_source.loc[annual_source["Year"] == anchor_year, "Month"].max())
            default_current_end = min(
                latest_anchor_month,
                calendar_today.month if anchor_year == calendar_today.year else latest_anchor_month,
            )
            default_current_period = (1, max(1, default_current_end))
            default_prior_period = default_current_period
            reset_invalid_month_range("yearly_current_period", default_current_period)
            reset_invalid_month_range("yearly_prior_period", default_prior_period)

            control_left, control_right, control_mode = st.columns([1, 1, 0.9])
            with control_left:
                current_period = st.select_slider(
                    f"{anchor_year} period",
                    options=list(range(1, 13)),
                    value=st.session_state["yearly_current_period"],
                    key="yearly_current_period",
                    format_func=month_name,
                )
            with control_right:
                if prior_year is not None:
                    prior_period = st.select_slider(
                        f"{prior_year} period",
                        options=list(range(1, 13)),
                        value=st.session_state["yearly_prior_period"],
                        key="yearly_prior_period",
                        format_func=month_name,
                    )
                else:
                    prior_period = default_prior_period
                    st.info("No prior year is available for comparison.")
            with control_mode:
                use_absolute_yearly = st.toggle(
                    "Absolute totals",
                    value=False,
                    help=(
                        "Off: compare years using average monthly spend. "
                        "On: compare period totals using the selected month windows."
                    ),
                )

            if use_absolute_yearly:
                annual_basis = annual_source[in_month_range(annual_source, current_period)].copy()
                annual_total = (
                    annual_basis.groupby("Year", as_index=False)["amount"].sum()
                    .rename(columns={"amount": "Amount"})
                    .sort_values("Year")
                )
                annual_cat = (
                    annual_basis.groupby(["Year", "category"], as_index=False)["amount"].sum()
                    .rename(columns={"category": "Category", "amount": "Amount"})
                )
                amount_label = f"Period total ({month_name(current_period[0])}-{month_name(current_period[1])})"
                hover_label = "Period total"
            else:
                active_months = annual_source.groupby("Year")["Month"].nunique().rename("Month count")
                annual_total = (
                    annual_source.groupby("Year", as_index=False)["amount"].sum()
                    .rename(columns={"amount": "Raw total"})
                    .sort_values("Year")
                    .merge(active_months, on="Year", how="left")
                )
                annual_total["Amount"] = annual_total["Raw total"] / annual_total["Month count"].clip(lower=1)
                annual_cat = (
                    annual_source.groupby(["Year", "category"], as_index=False)["amount"].sum()
                    .rename(columns={"category": "Category", "amount": "Raw amount"})
                    .merge(active_months, on="Year", how="left")
                )
                annual_cat["Amount"] = annual_cat["Raw amount"] / annual_cat["Month count"].clip(lower=1)
                amount_label = "Avg monthly spend"
                hover_label = "Monthly average"

            annual_total["Year"] = annual_total["Year"].astype(str)
            annual_cat["Year"] = annual_cat["Year"].astype(str)
            annual_total["YoY change"] = annual_total["Amount"].diff()
            annual_total["YoY %"] = annual_total["Amount"].pct_change()

            top_annual_categories = (
                annual_cat.groupby("Category", as_index=False)["Amount"].sum()
                .sort_values("Amount", ascending=False)
                .head(top_n)["Category"]
                .tolist()
            )
            annual_cat_plot = annual_cat.copy()
            annual_cat_plot["Category"] = annual_cat_plot["Category"].where(
                annual_cat_plot["Category"].isin(top_annual_categories),
                "Other",
            )
            annual_cat_plot = annual_cat_plot.groupby(["Year", "Category"], as_index=False)["Amount"].sum()
            yearly_color_map = {**category_color_map, "Other": "#9aa4b2"}

            top_driver = (
                annual_cat.sort_values(["Year", "Amount"], ascending=[True, False])
                .groupby("Year", as_index=False)
                .head(1)
                .rename(columns={"Category": "Top driver", "Amount": "Driver spend"})
            )
            annual_summary = annual_total.merge(top_driver, on="Year", how="left")
            annual_summary["Top driver share"] = annual_summary["Driver spend"] / annual_summary["Amount"].replace(0, pd.NA)

            left, right = st.columns([1, 1.35])
            with left:
                st.markdown("#### Household Budget by Year")
                annual_fig = px.bar(
                    annual_total,
                    x="Year",
                    y="Amount",
                    text="Amount",
                    labels={"Amount": amount_label},
                )
                annual_fig.update_traces(
                    marker_color="#4e79a7",
                    texttemplate="$%{text:,.0f}",
                    textposition="outside",
                    hovertemplate=f"<b>%{{x}}</b><br>{hover_label}: $%{{y:,.2f}}<extra></extra>",
                )
                annual_fig.update_yaxes(tickprefix="$")
                st.plotly_chart(apply_plot_style(annual_fig, 450), use_container_width=True)

            with right:
                st.markdown("#### Main Annual Budget Drivers")
                annual_stack = px.bar(
                    annual_cat_plot,
                    x="Year",
                    y="Amount",
                    color="Category",
                    color_discrete_map=yearly_color_map,
                    labels={"Amount": amount_label},
                )
                annual_stack.update_yaxes(tickprefix="$")
                annual_stack = apply_plot_style(annual_stack, 500)
                annual_stack.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.18,
                        xanchor="left",
                        x=0,
                    ),
                    legend_title_text="Category",
                    margin=dict(l=20, r=20, t=22, b=110),
                )
                st.plotly_chart(annual_stack, use_container_width=True)

            display_summary = annual_summary[
                ["Year", "Amount", "YoY change", "YoY %", "Top driver", "Driver spend", "Top driver share"]
            ].copy()
            display_summary = display_summary.rename(columns={"Amount": amount_label})
            st.dataframe(
                style_currency(display_summary.style, [amount_label, "YoY change", "Driver spend"]).format(
                    {"YoY %": "{:.1%}", "Top driver share": "{:.1%}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

            if prior_year is not None:
                current_period_df = annual_source[
                    (annual_source["Year"] == anchor_year) & in_month_range(annual_source, current_period)
                ]
                prior_period_df = annual_source[
                    (annual_source["Year"] == prior_year) & in_month_range(annual_source, prior_period)
                ]
                compare_rows = []
                for category in sorted(set(current_period_df["category"]).union(set(prior_period_df["category"]))):
                    current_amount = current_period_df.loc[current_period_df["category"] == category, "amount"].sum()
                    prior_amount = prior_period_df.loc[prior_period_df["category"] == category, "amount"].sum()
                    if not use_absolute_yearly:
                        current_amount = current_amount / month_count(current_period)
                        prior_amount = prior_amount / month_count(prior_period)
                    compare_rows.append({
                        "Category": category,
                        f"{prior_year}": prior_amount,
                        f"{anchor_year}": current_amount,
                        "Change": current_amount - prior_amount,
                    })

                st.subheader(f"{anchor_year} vs {prior_year} Category Drivers")
                if compare_rows:
                    comparison = pd.DataFrame(compare_rows)
                    comparison["Abs change"] = comparison["Change"].abs()
                    comparison = comparison.sort_values("Abs change", ascending=False)
                    comparison_top = comparison.head(top_n).sort_values("Change")
                    compare_metric = "period totals" if use_absolute_yearly else "monthly averages"

                    compare_fig = go.Figure(
                        go.Bar(
                            x=comparison_top["Change"],
                            y=comparison_top["Category"],
                            orientation="h",
                            marker_color=[
                                "#4e79a7" if value >= 0 else "#e15759"
                                for value in comparison_top["Change"]
                            ],
                            customdata=comparison_top[[f"{prior_year}", f"{anchor_year}"]].to_numpy(),
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                f"{prior_year}: $%{{customdata[0]:,.2f}}<br>"
                                f"{anchor_year}: $%{{customdata[1]:,.2f}}<br>"
                                "Change: $%{x:,.2f}<extra></extra>"
                            ),
                        ),
                    )
                    compare_fig.add_vline(x=0, line_color="#8290a3", line_width=1)
                    compare_fig.update_layout(
                        title=f"Largest changes by category ({compare_metric})",
                        xaxis_title=f"Positive = {anchor_year} higher",
                        yaxis_title=None,
                    )
                    compare_fig.update_xaxes(tickprefix="$")
                    st.plotly_chart(apply_plot_style(compare_fig, 460), use_container_width=True)

                    comparison_table = comparison.drop(columns=["Abs change"])
                    st.dataframe(
                        style_currency(comparison_table.style, [f"{prior_year}", f"{anchor_year}", "Change"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No spending exists in the selected comparison periods.")

    with categories_tab:
        if len(person_order) >= 2:
            p1, p2 = person_order[:2]
            chart_src = diff_table.head(top_n).sort_values("Difference")
            colors = chart_src["Difference"].apply(lambda v: person_color_map[p1] if v >= 0 else person_color_map[p2])
            diff_fig = go.Figure(
                go.Bar(
                    x=chart_src["Difference"],
                    y=chart_src["Category"],
                    orientation="h",
                    marker_color=colors,
                    customdata=chart_src[[p1, p2, "Higher spender"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        f"{p1}: $%{{customdata[0]:,.2f}}<br>"
                        f"{p2}: $%{{customdata[1]:,.2f}}<br>"
                        "Higher: %{customdata[2]}<br>"
                        "Gap: $%{x:,.2f}<extra></extra>"
                    ),
                )
            )
            diff_fig.add_vline(x=0, line_color="#8290a3", line_width=1)
            diff_fig.update_layout(
                title=f"Category Gaps: Positive = {p1} More, Negative = {p2} More",
                xaxis_title=f"Positive = {p1} spent more",
                yaxis_title=None,
            )
            diff_fig.update_xaxes(tickprefix="$")
            st.plotly_chart(apply_plot_style(diff_fig, 520), use_container_width=True)

            share_cols = [f"{p1} share", f"{p2} share"]
            display_cols = ["Category", p1, p2, "Difference", "Abs Difference", "Higher spender"] + share_cols
            table = diff_table[display_cols].copy()
            st.dataframe(
                style_currency(table.style, [p1, p2, "Difference", "Abs Difference"]).format(
                    {share_cols[0]: "{:.1%}", share_cols[1]: "{:.1%}"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Category differences need at least two people in the data.")

        mix = (
            lf.groupby(["person", "category"], as_index=False)["amount"].sum()
            .rename(columns={"person": "Person", "category": "Category", "amount": "Amount"})
        )
        mix["Share"] = mix["Amount"] / mix.groupby("Person")["Amount"].transform("sum")
        mix_top = mix[mix["Category"].isin(diff_table.head(top_n)["Category"].tolist())]
        mix_fig = px.bar(
            mix_top,
            x="Category",
            y="Share",
            color="Person",
            barmode="group",
            color_discrete_map=person_color_map,
            category_orders={"Person": person_order},
            title="Category Share of Each Person's Budget",
        )
        mix_fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(apply_plot_style(mix_fig, 430), use_container_width=True)

    with trends_tab:
        if len(person_order) >= 2:
            p1, p2 = person_order[:2]
            monthly_delta = (
                monthly_person.pivot(index="Month", columns="Person", values="Amount")
                .reindex(columns=person_order, fill_value=0.0)
                .fillna(0.0)
            )
            monthly_delta["Difference"] = monthly_delta[p1] - monthly_delta[p2]
            delta_fig = go.Figure(
                go.Bar(
                    x=monthly_delta.index,
                    y=monthly_delta["Difference"],
                    marker_color=[
                        person_color_map[p1] if value >= 0 else person_color_map[p2]
                        for value in monthly_delta["Difference"]
                    ],
                    hovertemplate="%{x|%Y-%m}<br>Gap: $%{y:,.2f}<extra></extra>",
                )
            )
            delta_fig.add_hline(y=0, line_color="#8290a3", line_width=1)
            delta_fig.update_layout(title=f"Monthly Difference: {p1} vs {p2}", yaxis_title="Gap")
            delta_fig.update_yaxes(tickprefix="$")
            st.plotly_chart(apply_plot_style(delta_fig, 400), use_container_width=True)

            heat_src = (
                monthly_cat.pivot_table(index="Category", columns=["Month", "Person"], values="Amount", aggfunc="sum", fill_value=0.0)
            )
            heat_rows = []
            for category in sorted(monthly_cat["Category"].unique()):
                for month in months_window:
                    p1_amt = heat_src.get((month, p1), pd.Series(dtype=float)).get(category, 0.0)
                    p2_amt = heat_src.get((month, p2), pd.Series(dtype=float)).get(category, 0.0)
                    heat_rows.append({"Category": category, "Month": month.strftime("%Y-%m"), "Difference": p1_amt - p2_amt})
            heat_df = pd.DataFrame(heat_rows)
            if not heat_df.empty:
                heat_fig = px.imshow(
                    heat_df.pivot(index="Category", columns="Month", values="Difference").fillna(0.0),
                    color_continuous_scale=[person_color_map[p2], "#f4f7fb", person_color_map[p1]],
                    aspect="auto",
                    title=f"Category Gap Heatmap ({p1} positive, {p2} negative)",
                    labels=dict(color="Gap"),
                )
                heat_fig.update_layout(coloraxis_colorbar=dict(tickprefix="$"))
                st.plotly_chart(apply_plot_style(heat_fig, max(420, 28 * heat_df["Category"].nunique())), use_container_width=True)
        else:
            st.info("Trend differences need at least two people in the data.")

        daily = (
            lf.groupby([pd.Grouper(key="date", freq="D"), "person"], as_index=False)["amount"].sum()
            .rename(columns={"date": "Date", "person": "Person", "amount": "Amount"})
        )
        full_days = pd.date_range(date_start, date_end, freq="D")
        daily_grid = pd.MultiIndex.from_product([full_days, person_order], names=["Date", "Person"])
        daily = daily.set_index(["Date", "Person"]).reindex(daily_grid, fill_value=0.0).reset_index()
        daily["Rolling 30-day spend"] = (
            daily.sort_values("Date")
            .groupby("Person")["Amount"]
            .rolling(30, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        rolling = px.line(
            daily,
            x="Date",
            y="Rolling 30-day spend",
            color="Person",
            color_discrete_map=person_color_map,
            category_orders={"Person": person_order},
            title="Rolling 30-Day Spend",
        )
        rolling.update_traces(line=dict(width=3))
        rolling.update_yaxes(tickprefix="$")
        st.plotly_chart(apply_plot_style(rolling, 420), use_container_width=True)

    with transactions_tab:
        table_df = lf.rename(
            columns={"date": "Date", "person": "Person", "category": "Category", "title": "Title", "amount": "Amount"}
        )[["Date", "Person", "Category", "Title", "Amount"]].sort_values(["Date", "Person"])
        table_df["Date"] = table_df["Date"].dt.date
        st.dataframe(
            style_currency(table_df.style, ["Amount"]),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Parsed preview", expanded=False):
            st.dataframe(long_df.sort_values("date").head(100), use_container_width=True, hide_index=True)
