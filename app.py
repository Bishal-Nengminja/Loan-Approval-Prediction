import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styles for consistent look & feel
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main { font-family: 'Inter', sans-serif; }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }

    /* Info Cards */
    .info-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }

    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    /* Prediction Box */
    .prediction-box {
        padding: 2rem;
        border-radius: 20px;
        margin: 2rem 0;
        text-align: center;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .prediction-box.approved {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        border: none;
    }
    .prediction-box.rejected {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        border: none;
    }

    /* Metrics */
    .custom-metric {
        background: linear-gradient(145deg, #f8fafc, #ffffff);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }

    /* Sidebar */
    .css-1d391kg { background: linear-gradient(180deg, #f8fafc, #ffffff); }

    /* Progress Bar */
    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_components():
    """Load model, scaler, and metadata with caching and error handling."""
    try:
        required_files = ['loan_approval_model.pkl', 'scaler.pkl', 'model_metadata.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            st.error("🚨 **Missing Model Files**")
            st.error(f"Cannot find: {', '.join(missing_files)}")
            st.info("**Solution:** Run the training notebook first to generate these files:")
            for file in required_files:
                st.code(file, language="text")
            st.stop()

        with open('loan_approval_model.pkl', 'rb') as f:
            model = pickle.load(f)
        scaler = joblib.load('scaler.pkl')
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        return model, scaler, metadata
    except Exception as e:
        st.error(f"❌ **Error loading model components:** {str(e)}")
        st.info("Please ensure all model files are generated correctly from the training notebook.")
        st.stop()


class LoanApprovalApp:
    def __init__(self):
        # Load model components from outside cached function
        self.model, self.scaler, self.metadata = load_model_components()

    def predict_loan_approval(self, input_data):
        """Make prediction with comprehensive validation"""
        try:
            if len(input_data) != len(self.metadata['features']):
                raise ValueError(f"Expected {len(self.metadata['features'])} features, got {len(input_data)}")

            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = self.scaler.transform(input_array)
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]  # [reject_prob, approve_prob]

            return prediction, probability
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None, None

    def create_feature_importance_chart(self):
        """Create feature importance chart"""
        try:
            importance_data = self.metadata['feature_importance']
            df = pd.DataFrame({
                'Feature': importance_data['Feature'],
                'Importance': importance_data['Importance'],
                'Abs_Importance': importance_data['Abs_Importance']
            }).sort_values('Abs_Importance', ascending=True)

            colors = ['#ef4444' if x < 0 else '#10b981' for x in df['Importance']]

            fig = go.Figure(go.Bar(
                y=df['Feature'],
                x=df['Importance'],
                orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.2)', width=1)),
                text=[f"{x:.3f}" for x in df['Importance']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': "🎯 Feature Importance in Loan Approval Prediction",
                    'font': {'size': 20, 'color': '#1f2937'},
                    'x': 0.5
                },
                xaxis_title="Importance Score",
                yaxis_title="Features",
                showlegend=False,
                height=500,
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
            return fig

        except Exception as e:
            st.error(f"Error creating feature importance chart: {str(e)}")
            return None

    def create_probability_gauge(self, probability):
        """Create probability gauge chart"""
        approval_prob = probability[1] * 100
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=approval_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "📊 Approval Probability", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "#10b981"}, 'decreasing': {'color': "#ef4444"}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#3b82f6", 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#fecaca'},
                    {'range': [25, 50], 'color': '#fed7aa'},
                    {'range': [50, 75], 'color': '#bfdbfe'},
                    {'range': [75, 100], 'color': '#bbf7d0'}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        fig.update_layout(height=400, font=dict(family="Inter, sans-serif"),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    def validate_input_data(self, input_data):
        """Validate input data with warnings"""
        warnings_list = []
        no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score, total_assets = input_data

        if cibil_score < 500:
            warnings_list.append("⚠️ Very low CIBIL score - approval chances are minimal")
        elif cibil_score < 650:
            warnings_list.append("⚠️ Below average CIBIL score - consider improving it")

        if loan_amount > income_annum * 10:
            warnings_list.append("⚠️ Loan amount is very high compared to annual income")

        if income_annum < 200000:
            warnings_list.append("⚠️ Low annual income - consider increasing income sources")

        if total_assets < loan_amount * 0.1:
            warnings_list.append("⚠️ Low asset value compared to loan amount")

        return warnings_list


def individual_prediction_page(app: LoanApprovalApp):
    st.markdown('<h2 class="sub-header">🔮 Individual Loan Prediction</h2>', unsafe_allow_html=True)

    # Step 1: Application Details
    st.markdown("""
    <div class="progress-bar"><div class="progress-fill" style="width: 33%"></div></div>
    <p style="text-align: center; color: #6b7280; font-size: 0.9rem;">Step 1 of 3: Enter Application Details</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 👤 Personal Information")
        no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5], help="Number of family members financially dependent on you")
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], help="Highest education qualification")
        self_employed = st.selectbox("Employment Type", ["No", "Yes"], help="Are you self-employed? (No = Salaried)")

        st.markdown("### 💰 Financial Information")
        income_annum = st.number_input("Annual Income (₹)", 100_000, 50_000_000, 500_000, step=50_000, help="Your total annual income in rupees")
        cibil_score = st.slider("CIBIL Score", 300, 900, 750, help="Your credit score (300-900). Higher is better!")

        # CIBIL score indicator
        if cibil_score >= 750:
            st.success("✅ Excellent credit score!")
        elif cibil_score >= 650:
            st.info("ℹ️ Good credit score")
        elif cibil_score >= 550:
            st.warning("⚠️ Average credit score")
        else:
            st.error("❌ Poor credit score")

    with col2:
        st.markdown("### 🏠 Loan Details")
        loan_amount = st.number_input("Loan Amount (₹)", 50_000, 50_000_000, 1_000_000, step=50_000, help="Amount you want to borrow")
        loan_term = st.slider("Loan Term (years)", 1, 30, 15, help="How many years to repay the loan")

        # EMI Calculator display
        if loan_amount and loan_term:
            monthly_rate = 0.08 / 12  # 8% annual interest assumed
            num_payments = loan_term * 12
            emi = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
            st.info(f"💡 **Estimated EMI:** ₹{emi:,.0f}/month")

        st.markdown("### 🏛️ Assets Information")
        residential_assets = st.number_input("Residential Property (₹)", 0, 100_000_000, 2_000_000, step=100_000)
        commercial_assets = st.number_input("Commercial Property (₹)", 0, 100_000_000, 0, step=100_000)
        luxury_assets = st.number_input("Luxury Assets (₹)", 0, 50_000_000, 500_000, step=50_000, help="Cars, jewelry, etc.")
        bank_assets = st.number_input("Bank Assets (₹)", 0, 50_000_000, 200_000, step=10_000, help="Savings, FD, investments")

    total_assets = residential_assets + commercial_assets + luxury_assets + bank_assets
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 0 if self_employed == "No" else 1

    # Step 2: Review Summary
    st.markdown("""
    <div class="progress-bar"><div class="progress-fill" style="width: 66%"></div></div>
    <p style="text-align: center; color: #6b7280; font-size: 0.9rem;">Step 2 of 3: Review Application Summary</p>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Application Summary")
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    sum_col1.markdown(f"""<div class="custom-metric"><div class="metric-value">₹{income_annum:,}</div><div class="metric-label">Annual Income</div></div>""", unsafe_allow_html=True)
    sum_col2.markdown(f"""<div class="custom-metric"><div class="metric-value">₹{loan_amount:,}</div><div class="metric-label">Loan Amount</div></div>""", unsafe_allow_html=True)
    sum_col3.markdown(f"""<div class="custom-metric"><div class="metric-value">{cibil_score}</div><div class="metric-label">CIBIL Score</div></div>""", unsafe_allow_html=True)
    sum_col4.markdown(f"""<div class="custom-metric"><div class="metric-value">₹{total_assets:,}</div><div class="metric-label">Total Assets</div></div>""", unsafe_allow_html=True)

    input_data = [
        no_of_dependents,
        education_encoded,
        self_employed_encoded,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        total_assets
    ]

    warnings_list = app.validate_input_data(input_data)
    if warnings_list:
        st.markdown("### ⚠️ Application Warnings")
        for warning in warnings_list:
            st.warning(warning)

    # Step 3: Prediction button
    st.markdown("---")
    _, pred_col, _ = st.columns([1, 2, 1])
    with pred_col:
        if st.button("🚀 Predict Loan Approval", type="primary", use_container_width=True):
            st.markdown("""
            <div class="progress-bar"><div class="progress-fill" style="width: 100%"></div></div>
            <p style="text-align: center; color: #6b7280; font-size: 0.9rem;">Step 3 of 3: Processing Results</p>
            """, unsafe_allow_html=True)

            with st.spinner("🔄 Analyzing your application with AI..."):
                prediction, probability = app.predict_loan_approval(input_data)

                if prediction is not None:
                    st.markdown("### 🎯 Prediction Results")
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        if prediction == 1:
                            st.markdown(f"""
                            <div class="prediction-box approved">
                                <h2>🎉 LOAN APPROVED!</h2>
                                <p>Congratulations! Your application meets our approval criteria.</p>
                                <p>Approval Probability: <strong>{probability[1]:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="prediction-box rejected">
                                <h2>❌ LOAN REJECTED</h2>
                                <p>Unfortunately, your application doesn't meet our current criteria.</p>
                                <p>Approval Probability: <strong>{probability[1]:.1%}</strong></p>
                            </div>
                            """, unsafe_allow_html=True)

                    with res_col2:
                        gauge_fig = app.create_probability_gauge(probability)
                        if gauge_fig:
                            st.plotly_chart(gauge_fig, use_container_width=True)

                    # Detailed analysis and recommendations
                    st.markdown("### 💡 Personalized Recommendations")
                    if prediction == 0:
                        st.markdown("""
                        <div class="info-card">
                            <ul style="text-align: left;">
                                <li><strong>Improve CIBIL Score:</strong> Pay your bills timely and reduce credit utilization</li>
                                <li><strong>Increase Income:</strong> Augment your income or wait for salary increments</li>
                                <li><strong>Reduce Loan Amount:</strong> Consider applying for a smaller loan</li>
                                <li><strong>Build Asset Base:</strong> Accumulate assets for better collateral</li>
                                <li><strong>Extend Loan Term:</strong> Longer terms reduce monthly EMI</li>
                                <li><strong>Enhance Education:</strong> Higher qualifications may boost approval chances</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-card">
                            <ul style="text-align: left;">
                                <li><strong>Strong Credit Profile</strong>: Your CIBIL score is favorable</li>
                                <li><strong>Adequate Income</strong>: Your income supports this loan</li>
                                <li><strong>Good Asset Base</strong>: Your assets secure the loan</li>
                                <li><strong>Low Risk Profile</strong>: Risk factors are minimal</li>
                            </ul>
                            <p><strong>Next Steps:</strong> Contact our loan officer to proceed with documentation!</p>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.info("Failed to get a prediction. Please try again.")


def model_analytics_page(app: LoanApprovalApp):
    st.markdown('<h2 class="sub-header">📊 Model Analytics & Insights</h2>', unsafe_allow_html=True)
    st.info("Model analytics page under construction.")


def batch_processing_page(app: LoanApprovalApp):
    st.markdown('<h2 class="sub-header">📁 Batch Processing</h2>', unsafe_allow_html=True)
    st.info("Batch processing page under construction.")


def about_page(app: LoanApprovalApp):
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    st.markdown("""
    This AI-powered Loan Approval System predicts loan approval probability using logistic regression.
    
    Developed with Streamlit, Plotly, and Scikit-learn.
    
    **Model trained on** loan application datasets with 8 key features.
    
    **For questions or support, contact:** support@loanai.com
    """)


def main():
    global app
    app = LoanApprovalApp()

    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 class="main-header">🏦 AI-Powered Loan Approval System</h1>
        <p style="font-size: 1.2rem; color: #6b7280; font-weight: 400;">
            Advanced machine learning predictions for instant loan decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        app_mode = st.selectbox("Choose Application Mode",
                                ["🔮 Individual Prediction", "📊 Model Analytics", "📁 Batch Processing", "ℹ️ About"], index=0)
        st.markdown("---")
        st.markdown("### 🎯 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{app.metadata['test_accuracy']:.1%}")
        col2.metric("Features", len(app.metadata['features']))
        st.markdown("---")
        st.markdown("### 📈 Quick Info")
        st.info("**Model Type:** Logistic Regression")
        st.info("**Training Features:** 8 key factors")
        st.info("**Last Updated:** " + datetime.now().strftime("%B %Y"))

        st.markdown("### 🔍 Feature Importance")
        fig = app.create_feature_importance_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    if app_mode == "🔮 Individual Prediction":
        individual_prediction_page(app)
    elif app_mode == "📊 Model Analytics":
        model_analytics_page(app)
    elif app_mode == "📁 Batch Processing":
        batch_processing_page(app)
    else:
        about_page(app)


if __name__ == "__main__":
    main()
