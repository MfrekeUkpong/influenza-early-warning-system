import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Public Health Influenza Early Warning System",
    page_icon="🦠",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA + MODEL
# --------------------------------------------------

model = joblib.load("../Models/random_forest_model.pkl")
thresholds = joblib.load("../Models/risk_thresholds.pkl")
data = pd.read_csv("../Outputs/weekly_influenza_model_data.csv")

data["WeekBeginning"] = pd.to_datetime(data["WeekBeginning"])
data = data.sort_values("WeekBeginning").reset_index(drop=True)

low_threshold = thresholds["low_threshold"]
high_threshold = thresholds["high_threshold"]


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def classify_risk(prediction, low, high):

    if prediction < low:
        return "Low Risk"

    elif prediction <= high:
        return "Medium Risk"

    return "High Risk"



def risk_colour(risk):

    if risk == "Low Risk":
        return "green"

    if risk == "Medium Risk":
        return "orange"

    return "red"



def show_risk_banner(risk):

    if risk == "Low Risk":
        st.success("🟢 Low Risk")

    elif risk == "Medium Risk":
        st.warning("🟠 Medium Risk")

    else:
        st.error("🔴 High Risk")



def risk_details(risk):

    if risk == "Low Risk":

        return dict(

            summary="Influenza activity is expected to remain relatively low.",

            warning="Current activity appears to remain within a lower expected range.",

            vulnerable="Routine awareness recommended for vulnerable populations including older adults and those with underlying conditions.",

            public_health="Routine surveillance monitoring can continue.",

            hospitals="No unusual hospital pressure expected."

        )


    if risk == "Medium Risk":

        return dict(

            summary="Moderate influenza activity expected. Increased monitoring advisable.",

            warning="Influenza activity may increase further and should be monitored closely.",

            vulnerable="Vulnerable populations should take additional precautions.",

            public_health="Strengthened surveillance and situational awareness recommended.",

            hospitals="Hospitals may review staffing levels and capacity planning."

        )


    return dict(

        summary="Possible influenza surge expected. Early preparedness may be needed.",

        warning="Possible influenza surge. Consider heightened monitoring and resource planning.",

        vulnerable="Higher risk expected among older adults, children and those with chronic illness.",

        public_health="Enhanced surveillance and communication strategies recommended.",

        hospitals="Hospitals may experience increased operational pressure."

    )


# --------------------------------------------------
# SHARED DATE LOGIC
# --------------------------------------------------

available_dates = sorted(data["WeekBeginning"].dt.date.unique())

default_date = available_dates[-1]


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("Public Health Influenza Early Warning System")

st.caption(
    "Prototype decision-support dashboard for influenza early warning in Scotland."
)


latest_data_date = data["WeekBeginning"].max().strftime("%d %b %Y")

st.caption(f"Latest surveillance data available: {latest_data_date}")


tab1, tab2 = st.tabs(["Surveillance Overview", "Prediction"])


# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.header("Historical Influenza Trend")

    picked_date = st.date_input(
        "Select Week Beginning Date",
        value=default_date,
        min_value=min(available_dates),
        max_value=max(available_dates)
    )


    selected_date = min(
        available_dates,
        key=lambda d: abs(d - picked_date)
    )


    selected_week_dt = pd.to_datetime(selected_date)

    forecast_week_dt = selected_week_dt + pd.Timedelta(weeks=2)


    selected_row = data[
        data["WeekBeginning"].dt.date == selected_date
    ].iloc[0]


    st.markdown(
        f"**Selected Week Beginning:** {selected_week_dt.strftime('%d %b %Y')}"
    )


    st.markdown(
        f"**Admissions:** {selected_row['admissions']:.0f}"
    )


    fig, ax = plt.subplots(figsize=(13,6))


    ax.plot(
        data["WeekBeginning"],
        data["admissions"],
        linewidth=2
    )


    ax.axvline(
        selected_week_dt,
        linestyle="--",
        linewidth=2
    )


    ax.axvline(
        forecast_week_dt,
        linestyle=":",
        linewidth=2
    )


    ax.set_title(
        "Weekly Influenza Admissions in Scotland"
    )


    ax.grid(True)


    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=6)
    )


    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %Y")
    )


    plt.xticks(rotation=45)


    st.pyplot(fig)


    st.caption(
        "Dashed line = selected surveillance week | Dotted line = forecast week"
    )


    st.subheader(
        "Recent Surveillance Data Around Selected Week"
    )


    recent_table = data[
        (data["WeekBeginning"] >= selected_week_dt - pd.Timedelta(weeks=6))
        &
        (data["WeekBeginning"] <= selected_week_dt + pd.Timedelta(weeks=6))
    ][["WeekBeginning","admissions"]]


    recent_table["WeekBeginning"] = recent_table["WeekBeginning"].dt.strftime("%d %b %Y")


    st.dataframe(
        recent_table,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.header(
        "Two-Week Ahead Influenza Admission Forecast and Risk Classification"
    )


    picked_date_pred = st.date_input(
        "Choose a week for prediction",
        value=default_date,
        min_value=min(available_dates),
        max_value=max(available_dates)
    )


    selected_date_pred = min(
        available_dates,
        key=lambda d: abs(d - picked_date_pred)
    )


    selected_week_dt_pred = pd.to_datetime(selected_date_pred)

    forecast_week_dt_pred = selected_week_dt_pred + pd.Timedelta(weeks=2)


    selected_row_pred = data[
        data["WeekBeginning"].dt.date == selected_date_pred
    ].iloc[0]


    lag_1 = selected_row_pred["lag_1"]

    lag_2 = selected_row_pred["lag_2"]

    lag_3 = selected_row_pred["lag_3"]


    if st.button("Predict Admissions and Risk"):


        prediction = float(
            model.predict(
                np.array([[lag_1, lag_2, lag_3]])
            )[0]
        )


        risk = classify_risk(
            prediction,
            low_threshold,
            high_threshold
        )


        details = risk_details(risk)


        st.subheader("Prediction Result")


        c1,c2,c3 = st.columns(3)


        c1.metric(
            "Selected Week",
            selected_week_dt_pred.strftime("%d %b %Y")
        )


        c2.metric(
            "Forecast Week",
            forecast_week_dt_pred.strftime("%d %b %Y")
        )


        c3.metric(
            "Predicted Admissions",
            f"{prediction:.2f}"
        )


        show_risk_banner(risk)


        st.caption(
            "Forecast generated using Random Forest using previous 3 weeks admissions."
        )


        st.subheader("Model Inputs Used")


        st.dataframe(

            pd.DataFrame({

                "Variable":[
                    "lag_1",
                    "lag_2",
                    "lag_3"
                ],

                "Admissions":[
                    lag_1,
                    lag_2,
                    lag_3
                ]

            }),

            hide_index=True,
            use_container_width=True
        )


        st.subheader("Recommended Actions")


        st.dataframe(

            pd.DataFrame({

                "Audience":[
                    "Vulnerable Groups",
                    "Public Health Teams",
                    "Hospitals"
                ],

                "Recommended Action":[
                    details["vulnerable"],
                    details["public_health"],
                    details["hospitals"]
                ]

            }),

            hide_index=True,
            use_container_width=True
        )


        st.subheader("Risk Thresholds")


        t1,t2,t3,t4 = st.columns([1,1,1,1.2])


        t1.metric("Low",f"< {low_threshold:.2f}")

        t2.metric("Medium",f"{low_threshold:.2f} – {high_threshold:.2f}")

        t3.metric("High",f"> {high_threshold:.2f}")


        report_df = pd.DataFrame({

            "Prediction":[prediction],

            "Risk":[risk]

        })


        t4.download_button(

            "Download Prediction Report",

            report_df.to_csv(index=False),

            "forecast.csv"

        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "This prototype demonstrates how short-term influenza admission forecasting can support early warning decision-making using Scottish surveillance data."
)