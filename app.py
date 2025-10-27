import streamlit as st
import pandas as pd
import plotly.express as px


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
    cols_lower = {c: c.strip().lower() for c in cols_list}

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

    # Extract names from headers (text after the prefix)
    def name_from_header(col: str, prefix: str) -> str:
        return col[len(prefix):].strip()

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
    person_order = [p for p in person_order if not (p in seen or seen.add(p))][:2]

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
    st.session_state["person_order"] = persons

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


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df_raw = pd.read_csv(uploaded_file)
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


st.title("Budget Timeline by Category")
st.caption("Upload your budget CSV to see per-category timelines for each person.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        long_df = load_uploaded_csv(uploaded_file)
    except Exception as e:
        st.error(str(e))
    else:
        # Clean up labels
        long_df["person"] = long_df["person"].map(_titlecase_person)
        long_df["category"] = long_df["category"].astype(str)

        st.subheader("Preview")
        st.dataframe(long_df.sort_values("date").head(50), use_container_width=True)

        # Month range filter (slider) — default to full window
        months_all = sorted(pd.to_datetime(long_df["date"]).dt.to_period("M").unique())
        month_labels = [str(m) for m in months_all]

        # Presets: All, This Year, Last 3 months
        if month_labels:
            default_range = (month_labels[0], month_labels[-1])
            if "month_range" not in st.session_state:
                st.session_state["month_range"] = default_range

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("All"):
                    st.session_state["month_range"] = default_range
            with c2:
                if st.button("This Year"):
                    current_year = pd.Timestamp.today().year
                    this_year_months = [m for m in months_all if m.year == current_year]
                    if this_year_months:
                        st.session_state["month_range"] = (str(this_year_months[0]), str(this_year_months[-1]))
                    else:
                        # If current year not present, fall back to last available year
                        if months_all:
                            last_year = months_all[-1].year
                            year_months = [m for m in months_all if m.year == last_year]
                            st.session_state["month_range"] = (str(year_months[0]), str(year_months[-1]))
            with c3:
                if st.button("Last 3 months"):
                    if len(months_all) >= 3:
                        last_three = months_all[-3:]
                        st.session_state["month_range"] = (str(last_three[0]), str(last_three[-1]))
                    elif months_all:
                        st.session_state["month_range"] = (str(months_all[0]), str(months_all[-1]))

            start_label, end_label = st.select_slider(
                "Month range",
                options=month_labels,
                value=st.session_state["month_range"],
                key="month_range",
            )
            start_period = pd.Period(start_label, freq="M")
            end_period = pd.Period(end_label, freq="M")
        else:
            # Fallback if no dates
            start_period = pd.Period("2000-01", freq="M")
            end_period = start_period

        date_start = start_period.start_time
        date_end = end_period.end_time

        # Controls
        categories = sorted(long_df["category"].unique())
        selected = st.multiselect("Select categories", categories, default=categories)
        cumulative = st.toggle("Show cumulative totals", value=True)

        # Apply month filter
        month_series = pd.to_datetime(long_df["date"]).dt.to_period("M")
        in_window = (month_series >= start_period) & (month_series <= end_period)
        lf = long_df[in_window].copy()

        # Aggregate by day per category/person and complete missing days in window
        base = (
            lf[lf["category"].isin(selected)]
            .groupby(["category", pd.Grouper(key="date", freq="D"), "person"], as_index=False)["amount"].sum()
            .rename(columns={"date": "Date", "category": "Category", "person": "Person", "amount": "Amount"})
        )

        # Complete daily grid for each Category/Person across selected window
        full_days = pd.date_range(date_start, date_end, freq="D")
        completed_frames = []
        for (cat, person), sub in base.groupby(["Category", "Person" ]):
            sub = sub.set_index("Date").reindex(full_days).fillna({"Amount": 0.0})
            sub["Category"] = cat
            sub["Person"] = person
            sub = sub.rename_axis("Date").reset_index()
            completed_frames.append(sub)
        df_plot = pd.concat(completed_frames, ignore_index=True) if completed_frames else base

        if cumulative:
            # Cumulative within each category/person
            df_plot["Amount"] = (
                df_plot.sort_values(["Category", "Person", "Date"])\
                    .groupby(["Category", "Person"])['Amount'].cumsum()
            )

        # Person colors and order for consistent legends
        if "person_order" in st.session_state and st.session_state["person_order"]:
            person_order = st.session_state["person_order"]
        else:
            person_order = sorted(long_df["person"].dropna().unique().tolist())
        # First person = red, second = blue, others fallback
        color_seq = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
        person_color_map = {p: color_seq[i % len(color_seq)] for i, p in enumerate(person_order)}

        # Overview: total per person over time (sum across categories)
        st.subheader("Timeline Overview")
        person_daily = (
            df_plot.groupby(["Date", "Person"], as_index=False)["Amount"].sum()
        )
        fig = px.line(
            person_daily,
            x="Date",
            y="Amount",
            color="Person",
            title=None,
            color_discrete_map=person_color_map,
            category_orders={"Person": person_order},
        )
        fig.update_xaxes(range=[date_start, date_end])
        st.plotly_chart(fig, use_container_width=True)

        # Separate charts per category with consistent window
        st.subheader("Per-Category Timelines")
        for cat in selected:
            sub = df_plot[df_plot["Category"] == cat]
            if sub.empty:
                continue
            fig_cat = px.line(
                sub,
                x="Date",
                y="Amount",
                color="Person",
                title=str(cat),
                color_discrete_map=person_color_map,
                category_orders={"Person": person_order},
            )
            fig_cat.update_xaxes(range=[date_start, date_end])
            st.plotly_chart(fig_cat, use_container_width=True)

            # Collapsible details table of the underlying costs for this category within the selected window
            detail = lf[lf["category"] == cat][["date", "person", "title", "amount"]].copy()
            if not detail.empty:
                detail = detail.rename(columns={
                    "date": "Date",
                    "person": "Person",
                    "title": "Title",
                    "amount": "Amount",
                })
                # Ensure datetime and sorting by Person then Date
                detail["Date"] = pd.to_datetime(detail["Date"])  # already datetime but safe
                # Remove time for display and save space
                detail["Date"] = detail["Date"].dt.date
                # Drop zero-amount rows
                detail = detail[detail["Amount"] != 0]
                # Order persons to preferred order
                persons_in = [p for p in person_order if p in detail["Person"].unique()]

                with st.expander(f"Details for {cat}", expanded=False):
                    cols = st.columns(len(persons_in) if persons_in else 1)
                    if persons_in:
                        for idx, p in enumerate(persons_in):
                            with cols[idx]:
                                st.markdown(f"<b style='color:{person_color_map.get(p, '#333')}'>{p}</b>", unsafe_allow_html=True)
                                dsub = detail[detail["Person"] == p].sort_values(["Date"])  
                                st.dataframe(
                                    dsub,
                                    use_container_width=True,
                                    hide_index=True,
                                )
                    else:
                        st.dataframe(detail.sort_values(["Person", "Date"]), hide_index=True)

        # Totals summary
        st.subheader("Totals by Person and Category")
        totals = long_df.groupby(["category", "person"], as_index=False)["amount"].sum()
        totals = totals.rename(columns={"category": "Category", "person": "Person", "amount": "Total"})
        st.dataframe(totals.sort_values(["Category", "Person"]), use_container_width=True)

        # Monthly pies per person (category proportions)
        st.subheader("Monthly Category Shares (Per Person)")
        # Persons to show (prefer consistent detected order)
        persons = person_order if person_order else sorted(long_df["person"].dropna().unique().tolist())

        # Color map: consistent category colors across months/persons
        all_cats = sorted(long_df["category"].dropna().astype(str).unique().tolist())
        palette = (
            px.colors.qualitative.Set3
            + px.colors.qualitative.Pastel
            + px.colors.qualitative.Safe
            + px.colors.qualitative.Vivid
        )
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(all_cats)}

        # Helper to render a donut pie with center text total
        def render_person_month_pie(df_sub: pd.DataFrame, person_label: str, month_label: str):
            if df_sub.empty:
                fig_empty = px.pie(names=["No Spend"], values=[1], hole=0.6)
                fig_empty.update_traces(textinfo="none", marker_colors=["#e0e0e0"]) 
                fig_empty.update_layout(
                    showlegend=False,
                    title=f"{person_label} — {month_label}",
                    annotations=[dict(text="$0", x=0.5, y=0.5, showarrow=False, font_size=16)],
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                return fig_empty

            by_cat = df_sub.groupby("category", as_index=False)["amount"].sum()
            total_amt = by_cat["amount"].sum()
            fig_pie = px.pie(
                by_cat,
                names="category",
                values="amount",
                color="category",
                color_discrete_map=color_map,
                hole=0.6,
                title=f"{person_label} — {month_label}",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(
                annotations=[dict(text=f"${total_amt:,.0f}", x=0.5, y=0.5, showarrow=False, font_size=16)],
                margin=dict(l=10, r=10, t=40, b=10),
            )
            return fig_pie

        # Build per-month rows with side-by-side pies for each person
        months = pd.to_datetime(long_df["date"]).dt.to_period("M").sort_values().unique()
        # Respect month range filter
        months = [m for m in months if (m >= start_period) and (m <= end_period)]
        for m in months:
            # Month label
            month_label = str(m)
            cols = st.columns(len(persons)) if len(persons) > 0 else [st.container()]
            for idx, person in enumerate(persons):
                with cols[idx]:
                    df_month_person = long_df[
                        (pd.to_datetime(long_df["date"]).dt.to_period("M") == m)
                        & (long_df["person"] == person)
                    ]
                    fig = render_person_month_pie(df_month_person, person, month_label)
                    st.plotly_chart(fig, use_container_width=True)

        st.success("Charts ready. Use the controls to explore.")
