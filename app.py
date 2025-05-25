import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import io
import sys
import os
import tempfile
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set page configuration
st.set_page_config(
    page_title="Media Mix Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add custom CSS for styling with dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #7FDBFF !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #7FDBFF !important;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .metric-container {
        background-color: rgba(40, 40, 40, 0.7);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("<h1 class='main-header'>Media Mix Modeling & Budget Optimization</h1>", unsafe_allow_html=True)

# Function to run the model and capture outputs
def run_model():
    """Run the call_model2.py script and capture its outputs and figures"""
    
    # Check if results are in session state to avoid rerunning
    if 'model_run' in st.session_state:
        return st.session_state['model_run']
    
    with st.spinner('Running model - this may take several minutes...'):
        # Path to the model script
        script_path = os.path.join(os.path.dirname(__file__), 'call_model2.py')
        
        # Create a temporary directory for saving figures
        with tempfile.TemporaryDirectory() as tmpdir:
            # Modify matplotlib to save figures instead of displaying them
            original_show = plt.show
            
            figures = []
            figure_data = {}
            
            # Override plt.show to save figures
            def custom_show():
                fig = plt.gcf()
                fig_num = len(figures)
                figures.append(fig)
                plt.savefig(f"{tmpdir}/figure_{fig_num}.png")
                plt.close(fig)
            
            plt.show = custom_show
            
            # Prepare to capture stdout and data variables
            output = io.StringIO()
            
            # Load the script as a module to run it and access its variables
            spec = importlib.util.spec_from_file_location("call_model2", script_path)
            model_module = importlib.util.module_from_spec(spec)
            
            # Redirect stdout to capture prints
            with redirect_stdout(output):
                try:
                    # Execute the module
                    spec.loader.exec_module(model_module)
                    
                    # Capture important data objects after execution
                    data = {
                        'data': getattr(model_module, 'data', None),
                        'X': getattr(model_module, 'X', None),
                        'y': getattr(model_module, 'y', None),
                        'y_pred': getattr(model_module, 'y_pred', None),
                        'dates': getattr(model_module, 'dates', None),
                        'media_channels': getattr(model_module, 'media_channels', None),
                        'final_model': getattr(model_module, 'final_model', None),
                        'forecast_df': getattr(model_module, 'forecast_df', None),
                        'adj_contributions': getattr(model_module, 'adj_contributions', None),
                        'response_df': getattr(model_module, 'response_df', None),
                        'spend_df': getattr(model_module, 'spend_df', None),
                        'actual_spend': getattr(model_module, 'actual_spend', None),
                        'optimal_spend': getattr(model_module, 'optimal_spend', None),
                        'optimal_spend_scaled': getattr(model_module, 'optimal_spend_scaled', None),
                    }
                    
                except Exception as e:
                    st.error(f"Error running the model: {str(e)}")
                    return None
            
            # Restore original plt.show function
            plt.show = original_show
            
            # Package the results
            results = {
                'figures': figures,
                'output_text': output.getvalue(),
                'data': data,
            }
            
            # Store in session state to avoid rerunning
            st.session_state['model_run'] = results
            return results

# Run button to execute the model
if st.button("Run Media Mix Model", key="run_model_btn") or 'model_run' in st.session_state:
    results = run_model()
    if results:
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Model Performance", "Media Contributions", "Budget Optimization", "Prophet Forecast"])
        
        with tab1:
            st.markdown("<h2 class='section-header'>Model Performance</h2>", unsafe_allow_html=True)
            # Recreate Actual vs Predicted plot with Plotly
            dates = results['data']['dates']
            y = results['data']['y'].values.flatten()
            y_pred = results['data']['y_pred'] 
            if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
                y_pred = y_pred.flatten()         
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, 
                y=y, 
                mode='lines+markers', 
                name='Actual Revenue', 
                line=dict(color='#00BFFF', width=2)  # Bright blue for dark background
            ))
            fig.add_trace(go.Scatter(
                x=dates, 
                y=y_pred, 
                mode='lines+markers', 
                name='Predicted Revenue', 
                line=dict(color='#FF4500', width=2, dash='dash')  # Bright orange-red
            ))
            
            fig.update_layout(
                title='Actual vs Predicted Revenue',
                xaxis_title='Date',
                yaxis_title='Revenue',
                legend_title='Legend',
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("<h2 class='section-header'>Media Channel Contributions</h2>", unsafe_allow_html=True)
            
            # Recreate media contributions area chart
            media_channels = results['data']['media_channels']
            adj_contributions = results['data']['adj_contributions']
            
            # Convert to stacked area chart with Plotly
            contribution_data = adj_contributions[media_channels + ['Base']]
            fig = go.Figure()
            
            # Vibrant color palette for dark background
            colors = ['#FF9966', '#FF6B6B', '#4ECDC4', '#C7F464', '#81D8D0', '#9932CC']
            
            # Add traces for each channel in reverse order for proper stacking
            for i, channel in enumerate(reversed(contribution_data.columns)):
                color_idx = i % len(colors)
                fig.add_trace(go.Scatter(
                    x=dates, 
                    y=contribution_data[channel],
                    mode='lines',
                    stackgroup='one',
                    name=channel,
                    hoverinfo='x+y',
                    line=dict(width=0.5, color=colors[color_idx])
                ))
            
            fig.update_layout(
                title='Revenue Contribution by Channel Over Time',
                xaxis_title='Date',
                yaxis_title='Revenue Contribution',
                hovermode='x unified',
                legend_title='Channels',
                template='plotly_dark',
                height=600,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Share of Spend vs Share of Effect
            st.markdown("<h3 class='section-header'>Effectiveness Analysis</h3>", unsafe_allow_html=True)
            
            response_df = results['data']['response_df']
            spend_df = results['data']['spend_df']
            
            # Combine the data
            comparison_df = pd.merge(response_df, spend_df, on='media')
            comparison_df = comparison_df.sort_values('effect_share', ascending=False)
            
            # Create the horizontal bar chart with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=comparison_df['media'],
                x=comparison_df['spend_share'],
                name='Share of Spend',
                orientation='h',
                marker=dict(color='rgba(0, 191, 255, 0.9)'),  # Bright blue for dark background
                text=[f'{x:.1f}%' for x in comparison_df['spend_share']],
                textposition='outside'
            ))
            
            fig.add_trace(go.Bar(
                y=comparison_df['media'],
                x=comparison_df['effect_share'],
                name='Share of Effect',
                orientation='h',
                marker=dict(color='rgba(255, 165, 0, 0.9)'),  # Orange for dark background
                text=[f'{x:.1f}%' for x in comparison_df['effect_share']],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Share of Effect vs Share of Spend by Channel',
                xaxis_title='Percentage (%)',
                yaxis_title='Media Channels',
                barmode='group',
                bargap=0.15,
                bargroupgap=0.1,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ROI Analysis
            st.markdown("<h3 class='section-header'>ROI Analysis</h3>", unsafe_allow_html=True)
            
            # Calculate ROI (Revenue generated per dollar spent)
            roi_df = comparison_df.copy()
            roi_df['roi'] = roi_df['total_effect'] / roi_df['total_spend']
            roi_df = roi_df.sort_values('roi', ascending=False)
            
            fig = px.bar(
                roi_df,
                x='media',
                y='roi',
                text=[f'{x:.2f}' for x in roi_df['roi']],
                color='roi',
                color_continuous_scale='Turbo',  # Vibrant color scale that works well on dark backgrounds
                title='Return on Investment (ROI) by Channel'
            )
            
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            fig.update_layout(
                xaxis_title='Media Channel',
                yaxis_title='ROI (Revenue per $ Spent)',
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("<h2 class='section-header'>Budget Optimization</h2>", unsafe_allow_html=True)
            
            # Prepare data for visualization
            actual_spend = results['data']['actual_spend']
            optimal_spend = results['data']['optimal_spend']
            optimal_spend_scaled = results['data']['optimal_spend_scaled']
            media_channels = results['data']['media_channels']
            
            # Create comparison dataframe
            budget_comparison = pd.DataFrame({
                'Channel': media_channels,
                'Actual Spend': actual_spend.values,
                'Optimal Spend': optimal_spend_scaled.values,
                'Change %': ((optimal_spend_scaled.values - actual_spend.values) / actual_spend.values * 100)
            })
            
            # Display summary statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Budget Summary")                
                st.markdown("</div>", unsafe_allow_html=True)
            
# Replace the projected revenue calculation section in tab3

            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.subheader("Optimization Impact")
                
                # Calculate projected revenue based on optimal allocation
                current_revenue = results['data']['y'].sum()[0]
                
                # Get the model and contribution data
                final_model = results['data']['final_model']
                media_channels = results['data']['media_channels']
                
                # Create a copy of the original data to simulate optimal spending
                X_optimal = results['data']['X'].copy()
                
                # Replace the media channel values with optimal values
                for i, channel in enumerate(media_channels):
                    X_optimal[channel] = optimal_spend_scaled[i]
                
                # Predict revenue with optimal allocation
                try:
                    # Use the model to predict based on optimal allocation
                    y_pred_optimal = final_model.predict(X_optimal)
                    projected_revenue = y_pred_optimal.sum()
                    
                    # Calculate the percentage increase
                    estimated_increase = (projected_revenue - current_revenue) / current_revenue
                except:
                    # Fallback calculation if model prediction fails
                    # Calculate based on ROI and spend difference
                    roi_df = comparison_df.copy()
                    roi_df['roi'] = roi_df['total_effect'] / roi_df['total_spend']
                    
                    # Calculate the additional revenue from spend changes
                    additional_revenue = 0
                    for i, channel in enumerate(media_channels):
                        spend_change = optimal_spend_scaled[i] - actual_spend[i]
                        channel_roi = roi_df.loc[roi_df['media'] == channel, 'roi'].values[0]
                        additional_revenue += spend_change * channel_roi
                    
                    projected_revenue = current_revenue + additional_revenue
                    estimated_increase = additional_revenue / current_revenue
                
                st.write(f"**Projected Revenue Increase:** {estimated_increase:.1%}")
                st.write(f"**Current Revenue:** {current_revenue:,.2f} INR")
                st.write(f"**Projected Revenue:** {projected_revenue:,.2f} INR")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display the budget comparison table
            st.markdown("<h3 class='section-header'>Budget Allocation Comparison</h3>", unsafe_allow_html=True)
            
            # Color formatting for the Change % column
            def color_negative_red(val):
                color = '#FF6B6B' if val < 0 else '#4ECDC4'  # Red for negative, teal for positive
                return f'color: {color}'
            
            st.dataframe(
                budget_comparison.style
                .format({'Actual Spend': '${:,.2f}', 'Optimal Spend': '${:,.2f}', 'Change %': '{:.1f}%'})
                .applymap(color_negative_red, subset=['Change %'])
            )
            
            # Visualization of budget allocation
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=budget_comparison['Channel'],
                y=budget_comparison['Actual Spend'],
                name='Current Allocation',
                marker_color='rgba(102, 204, 255, 0.9)',  # Light blue for dark backgrounds
                text=[f'${x:,.0f}' for x in budget_comparison['Actual Spend']],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                x=budget_comparison['Channel'],
                y=budget_comparison['Optimal Spend'],
                name='Optimal Allocation',
                marker_color='rgba(50, 255, 126, 0.9)',  # Bright green for dark backgrounds
                text=[f'${x:,.0f}' for x in budget_comparison['Optimal Spend']],
                textposition='auto'
            ))
            
            fig.update_layout(
                title='Current vs Optimal Budget Allocation',
                xaxis_title='Media Channel',
                yaxis_title='Spend Amount ($)',
                barmode='group',
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show percentage change chart
            fig = px.bar(
                budget_comparison,
                x='Channel',
                y='Change %',
                color='Change %',
                color_continuous_scale='RdBu_r',  # Reversed RdBu scale (blue for positive, red for negative)
                text=[f"{x:.1f}%" for x in budget_comparison['Change %']],
                title='Percentage Change in Budget Allocation'
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(
                xaxis_title='Media Channel',
                yaxis_title='Change in Allocation (%)',
                template='plotly_dark',
                height=500,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("<h2 class='section-header'>Future Revenue Forecast</h2>", unsafe_allow_html=True)
            
            # Display forecast with actual data
            forecast_df = results['data']['forecast_df']
            data = results['data']['data']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=data['revenue'],
                mode='lines+markers',
                name='Historical Revenue',
                line=dict(color='#00BFFF', width=2)  # Bright blue
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['yhat'],
                mode='lines+markers',
                name='Forecasted Revenue',
                line=dict(color='#32CD32', width=2, dash='dash')  # Bright green
            ))
            
            fig.update_layout(
                title='Revenue Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue',
                template='plotly_dark',
                height=600,
                paper_bgcolor='rgba(30, 30, 30, 0.8)',
                plot_bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add seasonality components if available
            if 'trend' in forecast_df.columns and 'season' in forecast_df.columns:
                st.markdown("<h3 class='section-header'>Forecast Components</h3>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trend component
                    fig = px.line(
                        forecast_df, 
                        x='date', 
                        y='trend',
                        title='Trend Component',
                        markers=True,
                        color_discrete_sequence=['#FF9966']  # Vibrant orange
                    )
                    fig.update_layout(
                        template='plotly_dark', 
                        height=400,
                        paper_bgcolor='rgba(30, 30, 30, 0.8)',
                        plot_bgcolor='rgba(30, 30, 30, 0.8)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Seasonality component
                    fig = px.line(
                        forecast_df, 
                        x='date', 
                        y='season',
                        title='Seasonal Component',
                        markers=True,
                        color_discrete_sequence=['#4ECDC4']  # Vibrant teal
                    )
                    fig.update_layout(
                        template='plotly_dark', 
                        height=400,
                        paper_bgcolor='rgba(30, 30, 30, 0.8)',
                        plot_bgcolor='rgba(30, 30, 30, 0.8)',
                        font=dict(color='white')
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to run the model. Please check the logs for details.")
else:
    st.info("Click the 'Run Media Mix Model' button to run the model and generate visualizations.")

# Footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #AAAAAA;">
    <p>Media Mix Modeling Dashboard | Created with Streamlit</p>
</div>
""", unsafe_allow_html=True)