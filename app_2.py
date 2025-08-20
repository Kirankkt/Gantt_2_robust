# app.py  — All-in-one Construction Manager
# - Keeps your existing public tables and features intact
# - Adds new finance/controls modules in a separate cm schema

import streamlit as st
import pandas as pd
import plotly.express as px
import io
from datetime import datetime, date
import sqlalchemy
from sqlalchemy import text
import time
import math
import json
from dateutil.relativedelta import relativedelta

# ============================================================
# GLOBAL CONFIG
# ============================================================
st.set_page_config(page_title="Construction Manager – All-in-One", layout="wide")

# New modules write here; original tables remain in public
SCHEMA_NEW = "cm"

def q(table: str) -> str:
    """Fully-qualified table name in the cm schema."""
    return f'{SCHEMA_NEW}.{table}'

# ============================================================
# SHARED HELPERS
# ============================================================
def convert_nan_to_none(data):
    """Recursively replace NaN values in data with None."""
    if isinstance(data, dict):
        return {k: convert_nan_to_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nan_to_none(item) for item in data]
    elif isinstance(data, float):
        return None if math.isnan(data) else data
    else:
        return data

@st.cache_resource
def get_engine():
    # uses the same secret you already have
    connection_string = st.secrets["postgres"]["connection_string"]
    engine = sqlalchemy.create_engine(connection_string)
    # ensure cm schema exists; keep search_path handy
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NEW}"))
        # Do NOT move public ahead of cm here; we’ll schema-qualify writes to be explicit.
    return engine

def exec_sql(sql: str, params: dict | None = None):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(sql), params or {})

def log_audit(action: str, table_name: str, old_data: dict = None):
    """Write to the existing public.timeline_audit table."""
    if old_data:
        processed_data = convert_nan_to_none(old_data)
        old_data_json = json.dumps(processed_data, default=str)
    else:
        old_data_json = None
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO public.timeline_audit (table_name, action, old_data)
            VALUES (:table_name, :action, :old_data)
        """), {"table_name": table_name, "action": action, "old_data": old_data_json})

# ============================================================
# ONE-TIME DDL FOR NEW MODULES (in cm schema ONLY)
# ============================================================
DDL = [
    # Employees, timesheets, payroll runs
    f"""
    CREATE TABLE IF NOT EXISTS {q("payroll_employees")} (
        id BIGSERIAL PRIMARY KEY,
        employee_name TEXT,
        role TEXT,
        pay_type TEXT,           -- Hourly | Salary
        pay_rate NUMERIC,
        ot_multiplier NUMERIC DEFAULT 1.5,
        start_date DATE,
        status TEXT DEFAULT 'Active',
        bank_account TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("payroll_timesheets")} (
        id BIGSERIAL PRIMARY KEY,
        work_date DATE,
        employee_id BIGINT,
        job_name TEXT,
        cost_code TEXT,
        hours_regular NUMERIC,
        hours_ot NUMERIC,
        notes TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("payroll_runs")} (
        id BIGSERIAL PRIMARY KEY,
        period_start DATE,
        period_end DATE,
        total_gross NUMERIC,
        metadata JSONB
    )""",
    # Accounts Receivable
    f"""
    CREATE TABLE IF NOT EXISTS {q("ar_invoices")} (
        id BIGSERIAL PRIMARY KEY,
        invoice_no TEXT,
        client_name TEXT,
        project_name TEXT,
        invoice_type TEXT,            -- Progress | Final | T&M
        period_from DATE,
        period_to DATE,
        invoice_date DATE,
        due_date DATE,
        contract_value NUMERIC,
        this_period_amount NUMERIC,
        previous_billed NUMERIC,
        retainage_pct NUMERIC,
        status TEXT,                  -- Draft | Sent | Partially Paid | Paid | Void
        notes TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("ar_payments")} (
        id BIGSERIAL PRIMARY KEY,
        invoice_id BIGINT,
        payment_date DATE,
        amount NUMERIC,
        method TEXT,      -- UPI | Wire | Cheque | Cash | Other
        reference TEXT,
        notes TEXT
    )""",
    # Vendors & POs
    f"""
    CREATE TABLE IF NOT EXISTS {q("vendors")} (
        id BIGSERIAL PRIMARY KEY,
        vendor_name TEXT,
        contact TEXT,
        phone TEXT,
        email TEXT,
        gstin TEXT,
        address TEXT,
        notes TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("purchase_orders")} (
        id BIGSERIAL PRIMARY KEY,
        po_number TEXT,
        vendor_id BIGINT,
        project_name TEXT,
        po_date DATE,
        status TEXT,          -- Draft | Sent | Partially Received | Closed | Cancelled
        item TEXT,
        quantity NUMERIC,
        unit_cost NUMERIC,
        notes TEXT
    )""",
    # Change Orders
    f"""
    CREATE TABLE IF NOT EXISTS {q("change_orders")} (
        id BIGSERIAL PRIMARY KEY,
        co_number TEXT,
        project_name TEXT,
        description TEXT,
        status TEXT,            -- Proposed | Approved | Rejected | Voided
        cost_impact NUMERIC,
        days_impact INTEGER,
        requested_date DATE,
        approved_date DATE
    )""",
    # Cost codes, budgets, job costs
    f"""
    CREATE TABLE IF NOT EXISTS {q("cost_codes")} (
        id BIGSERIAL PRIMARY KEY,
        code TEXT,              -- e.g., 03 30 00
        division TEXT,          -- e.g., 03 Concrete
        description TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("budgets")} (
        id BIGSERIAL PRIMARY KEY,
        project_name TEXT,
        cost_code TEXT,
        budget_amount NUMERIC
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("job_costs")} (
        id BIGSERIAL PRIMARY KEY,
        project_name TEXT,
        cost_code TEXT,
        vendor TEXT,
        cost_type TEXT,        -- Labor | Material | Equip | Subcontract | Other
        cost_date DATE,
        amount NUMERIC,
        notes TEXT
    )""",
    # Jobs & WIP
    f"""
    CREATE TABLE IF NOT EXISTS {q("jobs")} (
        id BIGSERIAL PRIMARY KEY,
        project_name TEXT UNIQUE,
        contract_value NUMERIC,
        est_total_cost NUMERIC,
        start_date DATE,
        end_date DATE,
        retainage_pct NUMERIC
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("wip_snapshots")} (
        id BIGSERIAL PRIMARY KEY,
        snapshot_date DATE,
        project_name TEXT,
        pct_complete NUMERIC,
        cost_to_date NUMERIC,
        billed_to_date NUMERIC,
        recognized_revenue NUMERIC,
        over_under_billings NUMERIC
    )""",
    # Daily logs & equipment
    f"""
    CREATE TABLE IF NOT EXISTS {q("daily_logs")} (
        id BIGSERIAL PRIMARY KEY,
        log_date DATE,
        project_name TEXT,
        weather TEXT,
        crew_count INTEGER,
        safety_incidents INTEGER,
        notes TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("equipment_logs")} (
        id BIGSERIAL PRIMARY KEY,
        log_date DATE,
        equipment_name TEXT,
        project_name TEXT,
        hours_used NUMERIC,
        location TEXT,
        maintenance_needed BOOLEAN DEFAULT FALSE,
        notes TEXT
    )""",
    # RFIs & Submittals
    f"""
    CREATE TABLE IF NOT EXISTS {q("rfis")} (
        id BIGSERIAL PRIMARY KEY,
        rfi_number TEXT,
        title TEXT,
        project_name TEXT,
        date_sent DATE,
        due_date DATE,
        status TEXT,       -- Open | Answered | Closed
        question TEXT,
        response TEXT
    )""",
    f"""
    CREATE TABLE IF NOT EXISTS {q("submittals")} (
        id BIGSERIAL PRIMARY KEY,
        submittal_number TEXT,
        package TEXT,
        project_name TEXT,
        spec_section TEXT,  -- CSI section
        date_submitted DATE,
        date_approved DATE,
        status TEXT,        -- Pending | Approved | Revise/Resubmit | Rejected
        notes TEXT
    )""",
]
def bootstrap_new_schema():
    for stmt in DDL:
        exec_sql(stmt)
bootstrap_new_schema()

# ============================================================
# EXISTING (PUBLIC) TABLES — LOADERS & SAVERS (unchanged logic)
# ============================================================
@st.cache_data(ttl=60)
def load_timeline_data() -> pd.DataFrame:
    engine = get_engine()
    query = "SELECT * FROM public.construction_timeline_1"
    df = pd.read_sql(query, engine)
    df.columns = df.columns.str.strip()
    mapping = {
        "activity": "Activity",
        "item": "Item",
        "task": "Task",
        "room": "Room",
        "location": "Location",
        "notes": "Notes",
        "start_date": "Start Date",
        "end_date": "End Date",
        "status": "Status",
        "workdays": "Workdays"
    }
    df.rename(columns=mapping, inplace=True)
    if "Start Date" in df.columns:
        df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    if "End Date" in df.columns:
        df["End Date"] = pd.to_datetime(df["End Date"], errors="coerce")
    if "Progress" not in df.columns:
        df["Progress"] = 0.0
    df["Status"] = df["Status"].astype(str).fillna("Not Started")
    return df

@st.cache_data(ttl=60)
def load_items_data() -> pd.DataFrame:
    engine = get_engine()
    query = "SELECT * FROM public.items_order_1"
    df = pd.read_sql(query, engine)
    df.columns = df.columns.str.strip()
    mapping = {
        "item": "Item",
        "quantity": "Quantity",
        "order_status": "Order Status",
        "delivery_status": "Delivery Status",
        "notes": "Notes"
    }
    df.rename(columns=mapping, inplace=True)
    df["Item"] = df["Item"].astype(str)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0).astype(int)
    df["Order Status"] = df["Order Status"].astype(str)
    df["Delivery Status"] = df["Delivery Status"].astype(str)
    df["Notes"] = df["Notes"].astype(str)
    return df

def save_timeline_data(df: pd.DataFrame):
    engine = get_engine()
    # write explicitly to public
    df.to_sql("construction_timeline_1", engine, if_exists="replace", index=False, schema="public")
    load_timeline_data.clear()

def save_items_data(df: pd.DataFrame):
    engine = get_engine()
    df.to_sql("items_order_1", engine, if_exists="replace", index=False, schema="public")
    load_items_data.clear()

# ============================================================
# GENERIC LOAD/SAVE FOR NEW (cm) TABLES
# ============================================================
@st.cache_data(ttl=60)
def load_df(sql: str):
    eng = get_engine()
    return pd.read_sql(sql, eng)

def save_cm_table(df: pd.DataFrame, table: str):
    eng = get_engine()
    df.to_sql(table, eng, if_exists="replace", index=False, schema=SCHEMA_NEW)

# ============================================================
# NAVIGATION
# ============================================================
PAGES = [
    "📋 Main Timeline & Gantt",
    "📦 Items to Order",
    "🏦 Accounts Receivable (Invoices & Payments)",
    "💰 Payroll (Employees, Timesheets, Runs)",
    "🏷️ Vendors & Purchase Orders",
    "🧩 Cost Codes & Budgets",
    "📑 Change Orders",
    "📒 Job Costs & WIP",
    "📝 Daily Logs & Equipment",
    "❓ RFIs & 📦 Submittals",
]
page = st.sidebar.radio("Go to:", PAGES, index=0)

# ============================================================
# PAGE 1 — MAIN TIMELINE (original, intact)
# ============================================================
if page.startswith("📋"):
    st.title("Construction Project Manager Dashboard II")
    st.markdown(
        "This dashboard provides an overview of the construction project, including task snapshots, "
        "timeline visualization, and progress tracking. Use the sidebar to filter and update data."
    )

    # Hide the tooltips in st.data_editor
    st.markdown("""
    <style>
    [data-testid="stDataEditor"] [role="tooltip"] { visibility: hidden !important; }
    </style>
    """, unsafe_allow_html=True)

    # 4. MAIN TIMELINE: DATA EDITOR & ROW/COLUMN MANAGEMENT
    df_main = load_timeline_data()
    st.subheader("Update Task Information (Main Timeline)")

    with st.sidebar.expander("Row & Column Management (Main Timeline)"):
        st.markdown("*Delete a row by index*")
        delete_index = st.text_input("Enter row index to delete (main table)", value="")
        if st.button("Delete Row (Main)"):
            if delete_index.isdigit():
                idx = int(delete_index)
                if 0 <= idx < len(df_main):
                    df_main.drop(df_main.index[idx], inplace=True)
                    try:
                        save_timeline_data(df_main)
                        st.sidebar.success(f"Row {idx} deleted and saved.")
                        df_main = load_timeline_data()
                    except Exception as e:
                        st.sidebar.error(f"Error saving data: {e}")
                else:
                    st.sidebar.error("Invalid index.")
            else:
                st.sidebar.error("Please enter a valid integer index.")

        st.markdown("*Add a new column*")
        new_col_name = st.text_input("New Column Name (main table)", value="")
        new_col_type = st.selectbox("Column Type (main table)", ["string", "integer", "float", "datetime"])
        if st.button("Add Column (Main)"):
            if new_col_name and new_col_name not in df_main.columns:
                if new_col_type == "string":
                    df_main[new_col_name] = ""
                    df_main[new_col_name] = df_main[new_col_name].astype(object)
                elif new_col_type == "integer":
                    df_main[new_col_name] = 0
                elif new_col_type == "float":
                    df_main[new_col_name] = 0.0
                elif new_col_type == "datetime":
                    df_main[new_col_name] = pd.NaT
                try:
                    save_timeline_data(df_main)
                    st.sidebar.success(f"Column '{new_col_name}' added and saved.")
                    df_main = load_timeline_data()
                except Exception as e:
                    st.sidebar.error(f"Error saving data: {e}")
            else:
                st.sidebar.warning("Column already exists or invalid name.")

        st.markdown("*Delete a column*")
        col_to_delete = st.selectbox(
            "Select Column to Delete (main table)",
            options=[""] + list(df_main.columns),
            index=0
        )
        if st.button("Delete Column (Main)"):
            if col_to_delete and col_to_delete in df_main.columns:
                df_main.drop(columns=[col_to_delete], inplace=True)
                try:
                    save_timeline_data(df_main)
                    st.sidebar.success(f"Column '{col_to_delete}' deleted and saved.")
                    df_main = load_timeline_data()
                except Exception as e:
                    st.sidebar.error(f"Error saving data: {e}")
            else:
                st.sidebar.warning("Please select a valid column.")

    # Configure columns for the data editor (same as your original)
    column_config_main = {}
    if "id" in df_main.columns:
        column_config_main["id"] = st.column_config.NumberColumn(
            "ID", help="Unique Row ID (auto-increment from DB)", disabled=True
        )
    for col in ["Activity", "Item", "Task", "Room", "Location"]:
        if col in df_main.columns:
            column_config_main[col] = st.column_config.TextColumn(col, help=f"Enter or select a value for {col}.")
    if "Status" in df_main.columns:
        column_config_main["Status"] = st.column_config.SelectboxColumn(
            "Status", options=["Finished", "In Progress", "Not Started", "Delayed"], help="Status"
        )
    if "Progress" in df_main.columns:
        column_config_main["Progress"] = st.column_config.NumberColumn(
            "Progress", min_value=0, max_value=100, step=1, help="Progress %"
        )
    if "Start Date" in df_main.columns:
        column_config_main["Start Date"] = st.column_config.DateColumn("Start Date", help="Project start date")
    if "End Date" in df_main.columns:
        column_config_main["End Date"] = st.column_config.DateColumn("End Date", help="Project end date")

    df_main_original = load_timeline_data()
    if "id" in df_main_original.columns:
        df_main_original = df_main_original[["id"] + [c for c in df_main_original.columns if c != "id"]]

    edited_df_main = st.data_editor(
        df_main_original.copy(),
        column_config=column_config_main,
        use_container_width=True,
        num_rows="dynamic"
    )
    st.info("After editing, click 'Save Updates (Main Timeline)' then 'Refresh Data (Main Timeline)' to apply.")

    if st.button("Save Updates (Main Timeline)"):
        def normalize_status(x):
            if pd.isna(x) or str(x).strip().lower() in ["", "na", "null", "none"]:
                return "Not Started"
            return x
        edited_df_main["Status"] = edited_df_main["Status"].apply(normalize_status)
        edited_df_main.loc[edited_df_main["Status"].str.lower() == "finished", "Progress"] = 100

        # Audit deletions
        original_indices = set(df_main_original.index)
        edited_indices = set(edited_df_main.index)
        deleted_indices = original_indices - edited_indices
        if deleted_indices:
            for idx in deleted_indices:
                old_data = df_main_original.loc[idx].to_dict()
                log_audit(action="DELETE", table_name="public.construction_timeline_1", old_data=old_data)
            st.info(f"Audit log: {len(deleted_indices)} row(s) deletion recorded.")

        try:
            save_timeline_data(edited_df_main)
            st.success("Main timeline data successfully saved!")
        except Exception as e:
            st.error(f"Error saving main timeline: {e}")

    if st.button("Refresh Data (Main Timeline)"):
        load_timeline_data.clear()
        st.markdown(
            """
            <script>
            var queryParams = new URLSearchParams(window.location.search);
            queryParams.set("refresh", Date.now());
            window.location.search = queryParams.toString();
            </script>
            """,
            unsafe_allow_html=True
        )

    # Sidebar filters & Gantt (same behavior as before)
    st.sidebar.header("Filter Options (Main Timeline)")

    def norm_unique(df_input: pd.DataFrame, col: str):
        if col not in df_input.columns:
            return []
        return sorted(set(df_input[col].dropna().astype(str).str.lower().str.strip()))

    for key in ["activity_filter","item_filter","task_filter","room_filter","location_filter","status_filter"]:
        if key not in st.session_state:
            st.session_state[key] = []

    default_date_range = (
        edited_df_main["Start Date"].min() if "Start Date" in edited_df_main.columns and not edited_df_main["Start Date"].isnull().all() else datetime.today(),
        edited_df_main["End Date"].max() if "End Date" in edited_df_main.columns and not edited_df_main["End Date"].isnull().all() else datetime.today()
    )
    selected_date_range = st.sidebar.date_input("Filter Date Range", value=default_date_range, key="date_range")

    if st.sidebar.button("Clear Filters (Main)"):
        for key in ["activity_filter","item_filter","task_filter","room_filter","location_filter","status_filter"]:
            st.session_state[key] = []

    a_opts = norm_unique(edited_df_main, "Activity")
    selected_activity_norm = st.sidebar.multiselect("Filter by Activity", options=a_opts,
        default=st.session_state["activity_filter"], key="activity_filter")
    i_opts = norm_unique(edited_df_main, "Item")
    selected_item_norm = st.sidebar.multiselect("Filter by Item", options=i_opts,
        default=st.session_state["item_filter"], key="item_filter")
    t_opts = norm_unique(edited_df_main, "Task")
    selected_task_norm = st.sidebar.multiselect("Filter by Task", options=t_opts,
        default=st.session_state["task_filter"], key="task_filter")
    r_opts = norm_unique(edited_df_main, "Room")
    selected_room_norm = st.sidebar.multiselect("Filter by Room", options=r_opts,
        default=st.session_state["room_filter"], key="room_filter")
    l_opts = norm_unique(edited_df_main, "Location")
    selected_location_norm = st.sidebar.multiselect("Filter by Location", options=l_opts,
        default=st.session_state["location_filter"], key="location_filter")
    s_opts = norm_unique(edited_df_main, "Status")
    selected_statuses = st.sidebar.multiselect("Filter by Status", options=s_opts,
        default=st.session_state["status_filter"], key="status_filter")

    show_finished = st.sidebar.checkbox("Show Finished Tasks", value=True)
    color_by_status = st.sidebar.checkbox("Color-code Gantt Chart by Status", value=True)

    st.sidebar.markdown("*Refine Gantt Grouping*")
    group_by_room = st.sidebar.checkbox("Group by Room", value=False)
    group_by_item = st.sidebar.checkbox("Group by Item", value=False)
    group_by_task = st.sidebar.checkbox("Group by Task", value=False)
    group_by_location = st.sidebar.checkbox("Group by Location", value=False)

    df_filtered = edited_df_main.copy()
    for col in ["Activity", "Item", "Task", "Room", "Location", "Status"]:
        df_filtered[col + "_norm"] = df_filtered[col].astype(str).str.lower().str.strip()

    if selected_activity_norm:
        df_filtered = df_filtered[df_filtered["Activity_norm"].isin(selected_activity_norm)]
    if selected_item_norm:
        df_filtered = df_filtered[df_filtered["Item_norm"].isin(selected_item_norm)]
    if selected_task_norm:
        df_filtered = df_filtered[df_filtered["Task_norm"].isin(selected_task_norm)]
    if selected_room_norm:
        df_filtered = df_filtered[df_filtered["Room_norm"].isin(selected_room_norm)]
    if selected_location_norm:
        df_filtered = df_filtered[df_filtered["Location_norm"].isin(selected_location_norm)]
    if selected_statuses:
        df_filtered = df_filtered[df_filtered["Status_norm"].isin(selected_statuses)]
    if not show_finished:
        df_filtered = df_filtered[~df_filtered["Status_norm"].isin(["finished"])]

    normcols = [c for c in df_filtered.columns if c.endswith("_norm")]
    df_filtered.drop(columns=normcols, inplace=True, errors="ignore")

    if "Start Date" in df_filtered.columns and "End Date" in df_filtered.columns:
        if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
            srange, erange = selected_date_range
            srange = pd.to_datetime(srange); erange = pd.to_datetime(erange)
            df_filtered = df_filtered[(df_filtered["Start Date"] >= srange) & (df_filtered["End Date"] <= erange)]
        else:
            st.warning("Please select both a start and an end date.")

    # Gantt chart
    def create_gantt_chart(df_input: pd.DataFrame, color_by_status: bool = True):
        df_input = df_input.dropna(subset=["Start Date", "End Date"])
        needed = ["Start Date", "End Date", "Status", "Progress"]
        missing = [c for c in needed if c not in df_input.columns]
        if missing:
            return px.scatter(title=f"Cannot build Gantt: missing {missing}")
        if df_input.empty:
            return px.scatter(title="No data to display for Gantt")

        group_cols = ["Activity"]
        if group_by_room and "Room" in df_input.columns: group_cols.append("Room")
        if group_by_item and "Item" in df_input.columns: group_cols.append("Item")
        if group_by_task and "Task" in df_input.columns: group_cols.append("Task")
        if group_by_location and "Location" in df_input.columns: group_cols.append("Location")
        if not group_cols:
            return px.scatter(title="No group columns selected for Gantt")

        grouped = (df_input.groupby(group_cols, dropna=False)
                   .agg({"Start Date": "min","End Date": "max","Progress": "mean",
                         "Status": lambda s: list(s.dropna().astype(str))})
                   .reset_index())
        grouped.rename(columns={
            "Start Date": "GroupStart", "End Date": "GroupEnd",
            "Progress": "AvgProgress","Status": "AllStatuses"
        }, inplace=True)

        now = pd.Timestamp(datetime.today().date())
        def aggregated_status(st_list, avg_prog, start_dt, end_dt):
            all_lower = [str(x).lower().strip() for x in st_list]
            if all(s == "finished" for s in all_lower) or avg_prog >= 100: return "Finished"
            if end_dt < now and avg_prog < 100: return "Delayed"
            total_duration = (end_dt - start_dt).total_seconds() or 1
            delay_threshold = start_dt + pd.Timedelta(seconds=total_duration * 0.5)
            if now > delay_threshold and avg_prog == 0: return "Delayed"
            if "in progress" in all_lower:
                return "Just Started" if avg_prog == 0 else "In Progress"
            return "Not Started"

        segments = []
        for _, row in grouped.iterrows():
            label = " | ".join(str(row[g]) for g in group_cols)
            st_list = row["AllStatuses"]; start = row["GroupStart"]; end = row["GroupEnd"]; avgp = row["AvgProgress"]
            final_st = aggregated_status(st_list, avgp, start, end)
            if final_st == "In Progress" and 0 < avgp < 100:
                total_s = (end - start).total_seconds()
                done_s = total_s * (avgp / 100.0)
                done_end = start + pd.Timedelta(seconds=done_s)
                segments.append({"Group Label": label,"Start": start,"End": done_end,
                                 "Display Status": "In Progress (Completed part)","Progress": f"{avgp:.0f}%"})
                remain_pct = 100 - avgp
                segments.append({"Group Label": label,"Start": done_end,"End": end,
                                 "Display Status": "In Progress (Remaining part)","Progress": f"{remain_pct:.0f}%"})
            else:
                segments.append({"Group Label": label,"Start": start,"End": end,
                                 "Display Status": final_st,"Progress": f"{avgp:.0f}%"})
        gantt_df = pd.DataFrame(segments)
        if gantt_df.empty: return px.scatter(title="No data after grouping for Gantt")
        color_map = {
            "Not Started": "lightgray", "Just Started": "lightgreen",
            "In Progress (Completed part)": "darkblue", "In Progress (Remaining part)": "lightgray",
            "Finished": "green", "Delayed": "red"
        }
        fig = px.timeline(gantt_df, x_start="Start", x_end="End", y="Group Label",
                          text="Progress", color="Display Status", color_discrete_map=color_map)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Timeline", showlegend=True)
        return fig

    gantt_fig = create_gantt_chart(df_filtered, color_by_status=color_by_status)

    # KPIs & visuals
    total_tasks = len(edited_df_main)
    edited_df_main["Status"] = edited_df_main["Status"].astype(str).fillna("Not Started")
    finished_count = edited_df_main[edited_df_main["Status"].str.lower() == "finished"].shape[0]
    completion_pct = (finished_count / total_tasks * 100) if total_tasks else 0
    inprogress_count = edited_df_main[edited_df_main["Status"].str.lower() == "in progress"].shape[0]
    notstart_count = edited_df_main[edited_df_main["Status"].str.lower() == "not started"].shape[0]

    today_dt = pd.Timestamp(datetime.today().date())
    if "End Date" in df_filtered.columns:
        overdue_df = df_filtered[(df_filtered["End Date"] < today_dt) & (df_filtered["Status"].str.lower() != "finished")]
        overdue_count = overdue_df.shape[0]
    else:
        overdue_df = pd.DataFrame(); overdue_count = 0

    if "Activity" in df_filtered.columns:
        dist_table = df_filtered.groupby("Activity").size().reset_index(name="Task Count")
        dist_fig = px.bar(dist_table, x="Activity", y="Task Count", title="Task Distribution by Activity")
    else:
        dist_fig = px.bar(title="No 'Activity' column to show distribution.")

    if "Start Date" in df_filtered.columns:
        next7_df = df_filtered[(df_filtered["Start Date"] >= today_dt) & (df_filtered["Start Date"] <= today_dt + pd.Timedelta(days=7))]
    else:
        next7_df = pd.DataFrame()

    # Display dashboard
    st.header("Dashboard Overview (Main Timeline)")
    st.subheader("Current Tasks Snapshot"); st.dataframe(df_filtered)
    st.subheader("Project Timeline"); st.plotly_chart(gantt_fig, use_container_width=True)
    st.metric("Overall Completion (%)", f"{completion_pct:.1f}%"); st.progress(completion_pct / 100)

    st.markdown("#### Additional Insights")
    st.markdown(f"*Overdue Tasks:* {overdue_count}")
    if not overdue_df.empty:
        st.dataframe(overdue_df[["Activity", "Room", "Task", "Status", "End Date"]])

    st.markdown("*Task Distribution by Activity:*")
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown("*Upcoming Tasks (Next 7 Days):*")
    if not next7_df.empty: st.dataframe(next7_df[["Activity", "Room", "Task", "Start Date", "Status"]])
    else: st.info("No upcoming tasks in the next 7 days.")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Tasks", total_tasks)
    mcol2.metric("In Progress", inprogress_count)
    mcol3.metric("Finished", finished_count)
    mcol4.metric("Not Started", notstart_count)
    st.markdown("---")

# ============================================================
# PAGE 2 — ITEMS TO ORDER (original, intact)
# ============================================================
elif page.startswith("📦"):
    st.header("Items to Order")

    # Load original items
    df_items_original = load_items_data()
    if "id" in df_items_original.columns:
        df_items_original = df_items_original[["id"] + [col for col in df_items_original.columns if col != "id"]]

    for needed_col in ["Item", "Quantity", "Order Status", "Delivery Status", "Notes"]:
        if needed_col not in df_items_original.columns:
            df_items_original[needed_col] = ""

    df_items_original["Item"] = df_items_original["Item"].astype(str)
    df_items_original["Quantity"] = pd.to_numeric(df_items_original["Quantity"], errors="coerce").fillna(0).astype(int)
    df_items_original["Order Status"] = df_items_original["Order Status"].astype(str)
    df_items_original["Delivery Status"] = df_items_original["Delivery Status"].astype(str)
    df_items_original["Notes"] = df_items_original["Notes"].astype(str)

    items_col_config = {}
    if "id" in df_items_original.columns:
        items_col_config["id"] = st.column_config.NumberColumn("ID", help="Unique Row ID (auto-increment from DB)", disabled=True)
    items_col_config["Item"] = st.column_config.TextColumn("Item", help="Enter the name of the item.")
    items_col_config["Quantity"] = st.column_config.NumberColumn("Quantity", min_value=0, step=1, help="Enter the quantity required.")
    items_col_config["Order Status"] = st.column_config.SelectboxColumn("Order Status", options=["Ordered", "Not Ordered"], help="Choose if this item is ordered or not.")
    items_col_config["Delivery Status"] = st.column_config.SelectboxColumn("Delivery Status", options=["Delivered", "Not Delivered", "Delayed"], help="Delivery status of the item.")
    items_col_config["Notes"] = st.column_config.TextColumn("Notes", help="Enter any notes or remarks here.")

    edited_df_items = st.data_editor(
        df_items_original.copy(),
        column_config=items_col_config,
        use_container_width=True,
        num_rows="dynamic"
    )

    st.info("After editing, click 'Save Items Table' then 'Refresh Items Table' to apply your changes.")

    if st.button("Save Items Table"):
        # Audit deletions
        original_indices = set(df_items_original.index)
        edited_indices = set(edited_df_items.index)
        deleted_indices = original_indices - edited_indices
        if deleted_indices:
            for idx in deleted_indices:
                old_data = df_items_original.loc[idx].to_dict()
                log_audit(action="DELETE", table_name="public.items_order_1", old_data=old_data)
            st.info(f"Audit log: {len(deleted_indices)} row(s) deletion recorded for Items_Order_1.")
        try:
            edited_df_items["Quantity"] = pd.to_numeric(edited_df_items["Quantity"], errors="coerce").fillna(0).astype(int)
            save_items_data(edited_df_items)
            st.success("Items table successfully saved to the database!")
        except Exception as e:
            st.error(f"Error saving items table: {e}")

    if st.button("Refresh Items Table"):
        load_items_data.clear()
        st.markdown("""
        <script>
        var queryParams = new URLSearchParams(window.location.search);
        queryParams.set("refresh", Date.now());
        window.location.search = queryParams.toString();
        </script>
        """, unsafe_allow_html=True)

    csv_buffer = io.StringIO()
    edited_df_items.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Items Table as CSV",
        data=csv_buffer.getvalue(),
        file_name="Cleaned_Items_Table.csv",
        mime="text/csv"
    )

# ============================================================
# PAGE 3 — ACCOUNTS RECEIVABLE
# ============================================================
elif page.startswith("🏦"):
    st.header("Accounts Receivable")

    inv = load_df(f"SELECT * FROM {q('ar_invoices')} ORDER BY invoice_date NULLS LAST, id")
    if "retainage_pct" in inv.columns:
        inv["retainage_pct"] = pd.to_numeric(inv["retainage_pct"], errors="coerce")
    inv_cfg = {
        "invoice_no": st.column_config.TextColumn("Invoice #"),
        "client_name": st.column_config.TextColumn("Client"),
        "project_name": st.column_config.TextColumn("Project"),
        "invoice_type": st.column_config.SelectboxColumn("Type", options=["Progress","Final","T&M"]),
        "period_from": st.column_config.DateColumn("Period From"),
        "period_to": st.column_config.DateColumn("Period To"),
        "invoice_date": st.column_config.DateColumn("Invoice Date"),
        "due_date": st.column_config.DateColumn("Due Date"),
        "contract_value": st.column_config.NumberColumn("Contract Value"),
        "this_period_amount": st.column_config.NumberColumn("This Period"),
        "previous_billed": st.column_config.NumberColumn("Prev. Billed To-Date"),
        "retainage_pct": st.column_config.NumberColumn("Retainage %", min_value=0, max_value=20, step=0.5),
        "status": st.column_config.SelectboxColumn("Status", options=["Draft","Sent","Partially Paid","Paid","Void"]),
        "notes": st.column_config.TextColumn("Notes"),
    }
    inv_edit = st.data_editor(inv, use_container_width=True, num_rows="dynamic", column_config=inv_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Invoices"): 
        try: save_cm_table(inv_edit, "ar_invoices"); st.success("Invoices saved.")
        except Exception as e: st.error(f"Save failed: {e}")
    if c2.button("Refresh Invoices"): load_df.clear(); st.rerun()

    st.subheader("Payments (Receipts)")
    pay = load_df(f"SELECT * FROM {q('ar_payments')} ORDER BY payment_date NULLS LAST, id")
    pay_cfg = {
        "invoice_id": st.column_config.NumberColumn("Invoice ID"),
        "payment_date": st.column_config.DateColumn("Payment Date"),
        "amount": st.column_config.NumberColumn("Amount"),
        "method": st.column_config.SelectboxColumn("Method", options=["UPI","Wire","Cheque","Cash","Other"]),
        "reference": st.column_config.TextColumn("Reference"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    pay_edit = st.data_editor(pay, use_container_width=True, num_rows="dynamic", column_config=pay_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Payments"): 
        try: save_cm_table(pay_edit, "ar_payments"); st.success("Payments saved.")
        except Exception as e: st.error(f"Save failed: {e}")
    if c4.button("Refresh Payments"): load_df.clear(); st.rerun()

    st.subheader("A/R Aging")
    today = pd.Timestamp(date.today())
    inv_join = inv_edit.copy()
    try:
        paid = pay_edit.groupby("invoice_id")["amount"].sum().rename("paid_to_date").reset_index()
        inv_join = inv_join.merge(paid, left_on="id", right_on="invoice_id", how="left")
    except Exception:
        inv_join["paid_to_date"] = 0
    inv_join["paid_to_date"] = inv_join["paid_to_date"].fillna(0)
    inv_join["balance"] = inv_join["this_period_amount"].fillna(0) + inv_join["previous_billed"].fillna(0)
    inv_join["balance"] = inv_join["balance"] - inv_join["paid_to_date"]
    inv_join["due_date"] = pd.to_datetime(inv_join["due_date"], errors="coerce")
    inv_join["days_past_due"] = (today - inv_join["due_date"]).dt.days
    bins = [-10**9, -1, 0, 30, 60, 90, 10**9]
    labels = ["Not Due","Due Today","0-30","31-60","61-90","90+"]
    inv_join["bucket"] = pd.cut(inv_join["days_past_due"], bins=bins, labels=labels)
    aging = inv_join.groupby("bucket")["balance"].sum().reindex(labels, fill_value=0).reset_index()
    st.dataframe(aging)
    st.plotly_chart(px.bar(aging, x="bucket", y="balance", title="A/R Aging by Bucket"), use_container_width=True)

    st.markdown("**Progress Billing (AIA-style) Export**")
    if not inv_edit.empty:
        pick_idx = st.selectbox("Pick an invoice row to export", options=inv_edit.index.tolist())
        if st.button("Generate XLSX"):
            row = inv_edit.loc[pick_idx].to_dict()
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine="xlsxwriter") as xw:
                summary = pd.DataFrame([{
                    "Project": row.get("project_name",""),
                    "Client": row.get("client_name",""),
                    "Invoice #": row.get("invoice_no",""),
                    "Invoice Date": row.get("invoice_date",""),
                    "Period From": row.get("period_from",""),
                    "Period To": row.get("period_to",""),
                    "Contract Value": row.get("contract_value",0),
                    "Prev Billed To-Date": row.get("previous_billed",0),
                    "This Period": row.get("this_period_amount",0),
                    "Retainage %": row.get("retainage_pct",0),
                }])
                summary.to_excel(xw, index=False, sheet_name="G702_Summary")
                schedule = pd.DataFrame([{
                    "Cost Code": "", "Description": "", "Scheduled Value": "",
                    "Work This Period": "", "Total Completed & Stored To-Date": "", "Balance to Finish": ""
                }])
                schedule.to_excel(xw, index=False, sheet_name="G703_Schedule")
            st.download_button("Download AIA XLSX", data=out.getvalue(),
                               file_name=f"AIA_{row.get('invoice_no','invoice')}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============================================================
# PAGE 4 — PAYROLL
# ============================================================
elif page.startswith("💰"):
    st.header("Payroll")

    st.subheader("Employees")
    emp = load_df(f"SELECT * FROM {q('payroll_employees')} ORDER BY id")
    emp_cfg = {
        "employee_name": st.column_config.TextColumn("Employee"),
        "role": st.column_config.TextColumn("Role"),
        "pay_type": st.column_config.SelectboxColumn("Pay Type", options=["Hourly","Salary"]),
        "pay_rate": st.column_config.NumberColumn("Rate"),
        "ot_multiplier": st.column_config.NumberColumn("OT x", min_value=1.0, max_value=3.0, step=0.1),
        "start_date": st.column_config.DateColumn("Start"),
        "status": st.column_config.SelectboxColumn("Status", options=["Active","Inactive"]),
        "bank_account": st.column_config.TextColumn("Bank (masked)"),
    }
    emp_edit = st.data_editor(emp, use_container_width=True, num_rows="dynamic", column_config=emp_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Employees"):
        try: save_cm_table(emp_edit, "payroll_employees"); st.success("Employees saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Employees"): load_df.clear(); st.rerun()

    st.subheader("Timesheets")
    ts = load_df(f"SELECT * FROM {q('payroll_timesheets')} ORDER BY work_date NULLS LAST, id")
    ts_cfg = {
        "work_date": st.column_config.DateColumn("Date"),
        "employee_id": st.column_config.NumberColumn("Employee ID"),
        "job_name": st.column_config.TextColumn("Job/Project"),
        "cost_code": st.column_config.TextColumn("Cost Code"),
        "hours_regular": st.column_config.NumberColumn("Hours (Reg)"),
        "hours_ot": st.column_config.NumberColumn("Hours (OT)"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    ts_edit = st.data_editor(ts, use_container_width=True, num_rows="dynamic", column_config=ts_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Timesheets"):
        try: save_cm_table(ts_edit, "payroll_timesheets"); st.success("Timesheets saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh Timesheets"): load_df.clear(); st.rerun()

    st.subheader("Run Payroll (Preview)")
    ps = st.date_input("Period Start", value=date.today() - relativedelta(days=7))
    pe = st.date_input("Period End", value=date.today())
    ts_win = ts_edit.copy()
    ts_win["work_date"] = pd.to_datetime(ts_win["work_date"], errors="coerce")
    mask = (ts_win["work_date"] >= pd.Timestamp(ps)) & (ts_win["work_date"] <= pd.Timestamp(pe))
    ts_win = ts_win[mask]

    emp_rates = emp_edit.set_index("id")["pay_rate"].to_dict() if not emp_edit.empty else {}
    emp_ot = emp_edit.set_index("id")["ot_multiplier"].to_dict() if not emp_edit.empty else {}
    rows = []
    for _, r in ts_win.iterrows():
        eid = r.get("employee_id")
        rate = float(emp_rates.get(eid, 0) or 0)
        otm = float(emp_ot.get(eid, 1.5) or 1.5)
        reg = float(r.get("hours_regular") or 0) * rate
        otv = float(r.get("hours_ot") or 0) * rate * otm
        rows.append({"employee_id": eid, "regular_pay": reg, "ot_pay": otv, "gross": reg + otv})
    preview = pd.DataFrame(rows)
    if not preview.empty:
        st.dataframe(preview)
        st.metric("Total Gross", f"{preview['gross'].sum():,.2f}")
        if st.button("Commit Payroll Run"):
            payload = {"detail": rows, "period": {"start": str(ps), "end": str(pe)}}
            exec_sql(f"""INSERT INTO {q('payroll_runs')} (period_start, period_end, total_gross, metadata)
                         VALUES (:s,:e,:g,:m)""",
                     {"s": ps, "e": pe, "g": float(preview['gross'].sum()), "m": json.dumps(payload)})
            st.success("Payroll run stored.")

# ============================================================
# PAGE 5 — VENDORS & POs
# ============================================================
elif page.startswith("🏷️"):
    st.header("Vendors & Purchase Orders")

    st.subheader("Vendors")
    v = load_df(f"SELECT * FROM {q('vendors')} ORDER BY vendor_name")
    v_cfg = {
        "vendor_name": st.column_config.TextColumn("Vendor"),
        "contact": st.column_config.TextColumn("Contact"),
        "phone": st.column_config.TextColumn("Phone"),
        "email": st.column_config.TextColumn("Email"),
        "gstin": st.column_config.TextColumn("GSTIN"),
        "address": st.column_config.TextColumn("Address"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    v_edit = st.data_editor(v, use_container_width=True, num_rows="dynamic", column_config=v_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Vendors"):
        try: save_cm_table(v_edit, "vendors"); st.success("Vendors saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Vendors"): load_df.clear(); st.rerun()

    st.subheader("Purchase Orders")
    po = load_df(f"SELECT * FROM {q('purchase_orders')} ORDER BY po_date NULLS LAST, id")
    po_cfg = {
        "po_number": st.column_config.TextColumn("PO #"),
        "vendor_id": st.column_config.NumberColumn("Vendor ID"),
        "project_name": st.column_config.TextColumn("Project"),
        "po_date": st.column_config.DateColumn("PO Date"),
        "status": st.column_config.SelectboxColumn("Status", options=["Draft","Sent","Partially Received","Closed","Cancelled"]),
        "item": st.column_config.TextColumn("Item"),
        "quantity": st.column_config.NumberColumn("Qty"),
        "unit_cost": st.column_config.NumberColumn("Unit Cost"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    po_edit = st.data_editor(po, use_container_width=True, num_rows="dynamic", column_config=po_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save POs"):
        try: save_cm_table(po_edit, "purchase_orders"); st.success("POs saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh POs"): load_df.clear(); st.rerun()

# ============================================================
# PAGE 6 — COST CODES & BUDGETS
# ============================================================
elif page.startswith("🧩"):
    st.header("Cost Codes & Budgets")

    st.subheader("Cost Codes (CSI style)")
    cc = load_df(f"SELECT * FROM {q('cost_codes')} ORDER BY code")
    cc_cfg = {
        "code": st.column_config.TextColumn("Code (e.g., 03 30 00)"),
        "division": st.column_config.TextColumn("Division (e.g., 03 Concrete)"),
        "description": st.column_config.TextColumn("Description"),
    }
    cc_edit = st.data_editor(cc, use_container_width=True, num_rows="dynamic", column_config=cc_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Cost Codes"):
        try: save_cm_table(cc_edit, "cost_codes"); st.success("Cost codes saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Cost Codes"): load_df.clear(); st.rerun()

    st.subheader("Budgets")
    b = load_df(f"SELECT * FROM {q('budgets')} ORDER BY project_name, cost_code")
    b_cfg = {
        "project_name": st.column_config.TextColumn("Project"),
        "cost_code": st.column_config.TextColumn("Cost Code"),
        "budget_amount": st.column_config.NumberColumn("Budget Amount"),
    }
    b_edit = st.data_editor(b, use_container_width=True, num_rows="dynamic", column_config=b_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Budgets"):
        try: save_cm_table(b_edit, "budgets"); st.success("Budgets saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh Budgets"): load_df.clear(); st.rerun()

# ============================================================
# PAGE 7 — CHANGE ORDERS
# ============================================================
elif page.startswith("📑"):
    st.header("Change Orders")
    co = load_df(f"SELECT * FROM {q('change_orders')} ORDER BY requested_date NULLS LAST, id")
    co_cfg = {
        "co_number": st.column_config.TextColumn("CO #"),
        "project_name": st.column_config.TextColumn("Project"),
        "description": st.column_config.TextColumn("Description"),
        "status": st.column_config.SelectboxColumn("Status", options=["Proposed","Approved","Rejected","Voided"]),
        "cost_impact": st.column_config.NumberColumn("Cost Impact"),
        "days_impact": st.column_config.NumberColumn("Days Impact"),
        "requested_date": st.column_config.DateColumn("Requested"),
        "approved_date": st.column_config.DateColumn("Approved"),
    }
    co_edit = st.data_editor(co, use_container_width=True, num_rows="dynamic", column_config=co_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Change Orders"):
        try: save_cm_table(co_edit, "change_orders"); st.success("COs saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Change Orders"): load_df.clear(); st.rerun()

# ============================================================
# PAGE 8 — JOB COSTS & WIP
# ============================================================
elif page.startswith("📒"):
    st.header("Job Costs & WIP")

    st.subheader("Jobs (contract & estimates)")
    jobs = load_df(f"SELECT * FROM {q('jobs')} ORDER BY project_name")
    jobs_cfg = {
        "project_name": st.column_config.TextColumn("Project"),
        "contract_value": st.column_config.NumberColumn("Contract Value"),
        "est_total_cost": st.column_config.NumberColumn("Est. Total Cost"),
        "start_date": st.column_config.DateColumn("Start"),
        "end_date": st.column_config.DateColumn("End"),
        "retainage_pct": st.column_config.NumberColumn("Retainage %", min_value=0, max_value=20, step=0.5),
    }
    jobs_edit = st.data_editor(jobs, use_container_width=True, num_rows="dynamic", column_config=jobs_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Jobs"):
        try: save_cm_table(jobs_edit, "jobs"); st.success("Jobs saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Jobs"): load_df.clear(); st.rerun()

    st.subheader("Job Costs (actuals)")
    jc = load_df(f"SELECT * FROM {q('job_costs')} ORDER BY cost_date NULLS LAST, id")
    jc_cfg = {
        "project_name": st.column_config.TextColumn("Project"),
        "cost_code": st.column_config.TextColumn("Cost Code"),
        "vendor": st.column_config.TextColumn("Vendor/Subcontractor"),
        "cost_type": st.column_config.SelectboxColumn("Type", options=["Labor","Material","Equip","Subcontract","Other"]),
        "cost_date": st.column_config.DateColumn("Date"),
        "amount": st.column_config.NumberColumn("Amount"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    jc_edit = st.data_editor(jc, use_container_width=True, num_rows="dynamic", column_config=jc_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Job Costs"):
        try: save_cm_table(jc_edit, "job_costs"); st.success("Job costs saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh Job Costs"): load_df.clear(); st.rerun()

    st.subheader("WIP (Percent-of-Completion)")
    inv_df = load_df(f"SELECT * FROM {q('ar_invoices')}")
    costs_tot = jc_edit.groupby("project_name")["amount"].sum().rename("cost_to_date").reset_index()
    billed_tot = inv_df.groupby("project_name")["this_period_amount"].sum().rename("billed_to_date").reset_index()
    wip = jobs_edit.merge(costs_tot, on="project_name", how="left").merge(billed_tot, on="project_name", how="left")
    wip["cost_to_date"] = wip["cost_to_date"].fillna(0)
    wip["billed_to_date"] = wip["billed_to_date"].fillna(0)
    wip["pct_complete"] = (wip["cost_to_date"] / wip["est_total_cost"]).replace([pd.NA, pd.NaT, float("inf")], 0)
    wip["recognized_revenue"] = (wip["pct_complete"] * wip["contract_value"]).fillna(0)
    wip["over_under_billings"] = wip["billed_to_date"] - wip["recognized_revenue"]
    st.dataframe(wip)
    st.plotly_chart(px.bar(wip, x="project_name", y="over_under_billings",
                           title="Over/Under Billings by Project"), use_container_width=True)

    if st.button("Snapshot WIP Today"):
        rows = []
        for _, r in wip.iterrows():
            rows.append({
                "snapshot_date": date.today(),
                "project_name": r["project_name"],
                "pct_complete": float(r.get("pct_complete") or 0),
                "cost_to_date": float(r.get("cost_to_date") or 0),
                "billed_to_date": float(r.get("billed_to_date") or 0),
                "recognized_revenue": float(r.get("recognized_revenue") or 0),
                "over_under_billings": float(r.get("over_under_billings") or 0),
            })
        out = pd.DataFrame(rows)
        try:
            existing = load_df(f"SELECT * FROM {q('wip_snapshots')}")
            out = pd.concat([existing, out], ignore_index=True)
        except Exception:
            pass
        save_cm_table(out, "wip_snapshots")
        st.success("WIP snapshot stored.")

# ============================================================
# PAGE 9 — DAILY LOGS & EQUIPMENT
# ============================================================
elif page.startswith("📝"):
    st.header("Daily Logs & Equipment")

    st.subheader("Daily Logs")
    dl = load_df(f"SELECT * FROM {q('daily_logs')} ORDER BY log_date NULLS LAST, id")
    dl_cfg = {
        "log_date": st.column_config.DateColumn("Date"),
        "project_name": st.column_config.TextColumn("Project"),
        "weather": st.column_config.TextColumn("Weather"),
        "crew_count": st.column_config.NumberColumn("Crew #"),
        "safety_incidents": st.column_config.NumberColumn("Safety Incidents"),
        "notes": st.column_config.TextColumn("Notes / Visitors / Delays"),
    }
    dl_edit = st.data_editor(dl, use_container_width=True, num_rows="dynamic", column_config=dl_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save Daily Logs"):
        try: save_cm_table(dl_edit, "daily_logs"); st.success("Daily logs saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh Daily Logs"): load_df.clear(); st.rerun()

    st.subheader("Equipment Usage")
    eq = load_df(f"SELECT * FROM {q('equipment_logs')} ORDER BY log_date NULLS LAST, id")
    eq_cfg = {
        "log_date": st.column_config.DateColumn("Date"),
        "equipment_name": st.column_config.TextColumn("Equipment"),
        "project_name": st.column_config.TextColumn("Project"),
        "hours_used": st.column_config.NumberColumn("Hours Used"),
        "location": st.column_config.TextColumn("Location"),
        "maintenance_needed": st.column_config.CheckboxColumn("Maintenance Needed"),
        "notes": st.column_config.TextColumn("Notes"),
    }
    eq_edit = st.data_editor(eq, use_container_width=True, num_rows="dynamic", column_config=eq_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Equipment Logs"):
        try: save_cm_table(eq_edit, "equipment_logs"); st.success("Equipment logs saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh Equipment Logs"): load_df.clear(); st.rerun()

# ============================================================
# PAGE 10 — RFIs & SUBMITTALS
# ============================================================
elif page.startswith("❓"):
    st.header("RFIs & Submittals")

    st.subheader("RFIs (Requests for Information)")
    rfi = load_df(f"SELECT * FROM {q('rfis')} ORDER BY date_sent NULLS LAST, id")
    rfi_cfg = {
        "rfi_number": st.column_config.TextColumn("RFI #"),
        "title": st.column_config.TextColumn("Title"),
        "project_name": st.column_config.TextColumn("Project"),
        "date_sent": st.column_config.DateColumn("Date Sent"),
        "due_date": st.column_config.DateColumn("Due Date"),
        "status": st.column_config.SelectboxColumn("Status", options=["Open","Answered","Closed"]),
        "question": st.column_config.TextColumn("Question"),
        "response": st.column_config.TextColumn("Response"),
    }
    rfi_edit = st.data_editor(rfi, use_container_width=True, num_rows="dynamic", column_config=rfi_cfg)
    c1, c2 = st.columns(2)
    if c1.button("Save RFIs"):
        try: save_cm_table(rfi_edit, "rfis"); st.success("RFIs saved.")
        except Exception as e: st.error(e)
    if c2.button("Refresh RFIs"): load_df.clear(); st.rerun()

    st.subheader("Submittals")
    sub = load_df(f"SELECT * FROM {q('submittals')} ORDER BY date_submitted NULLS LAST, id")
    sub_cfg = {
        "submittal_number": st.column_config.TextColumn("Submittal #"),
        "package": st.column_config.TextColumn("Package"),
        "project_name": st.column_config.TextColumn("Project"),
        "spec_section": st.column_config.TextColumn("Spec (CSI Section)"),
        "date_submitted": st.column_config.DateColumn("Submitted"),
        "date_approved": st.column_config.DateColumn("Approved"),
        "status": st.column_config.SelectboxColumn("Status", options=["Pending","Approved","Revise/Resubmit","Rejected"]),
        "notes": st.column_config.TextColumn("Notes"),
    }
    sub_edit = st.data_editor(sub, use_container_width=True, num_rows="dynamic", column_config=sub_cfg)
    c3, c4 = st.columns(2)
    if c3.button("Save Submittals"):
        try: save_cm_table(sub_edit, "submittals"); st.success("Submittals saved.")
        except Exception as e: st.error(e)
    if c4.button("Refresh Submittals"): load_df.clear(); st.rerun()
