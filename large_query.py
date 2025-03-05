import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import warnings
import logging
from datetime import datetime, timedelta
import re
import nltk
from nltk.corpus import wordnet
import json
import sentencepiece

# Configure warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

class DatasetQueryAssistant:
    def __init__(self):
        self.initialize_page()
        self.initialize_model()
        self.initialize_intent_mapping()

    def initialize_page(self):
        """Initialize Streamlit page configuration"""
        st.set_page_config(
            page_title="Advanced Dataset Query Assistant",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("Advanced Dataset Query Assistant")
        st.sidebar.header("Configuration")

    def initialize_model(self):
        """Initialize FLAN-T5 model and tokenizer"""
        try:
            @st.cache_resource
            def load_model():
                device = "cpu"  # Simplify device handling
                model_name = "google/flan-t5-base"
                tokenizer =  T5Tokenizer.from_pretrained(model_name)
                model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Always use float32 for stability
                    device_map=None  # Let PyTorch handle device mapping
                )
                model = model.to(device)  # Explicitly move model to device
                return tokenizer, model, device

            self.tokenizer, self.model, self.device = load_model()
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    def initialize_intent_mapping(self):
        """Initialize intent mapping dictionary for query understanding"""
        # Map various synonyms and related words to their core intents
        self.intent_map = {
            # Statistical analysis
            "basic_statistics": ["basic stat", "basic statistics", "summary statistics", "descriptive statistics", 
                             "statistical summary", "stats", "describe", "summarize", "overview"],
            
            # Distribution analysis
            "distribution": ["distribution", "histogram", "frequency", "density", "spread"],
            
            # Box plot
            "boxplot": ["boxplot", "box plot", "box-plot", "box and whisker", "five-number summary", "outliers"],
            
            # Time series analysis
            "time_series": ["time series", "temporal", "trend", "trends", "time trend", "over time", "chronological"],
            
            # Time periods
            "hourly": ["hourly", "hour", "hours", "per hour"],
            "daily": ["daily", "day", "days", "per day"],
            "monthly": ["monthly", "month", "months", "per month"],
            "yearly": ["yearly", "year", "years", "per year", "annual"],
            "weekday": ["weekday", "weekdays", "working day"],
            "weekend": ["weekend", "weekends", "sat", "sun", "saturday", "sunday"],
            
            # Time comparisons
            "compare_halves": ["compare halves", "first half", "second half", "two halves", "half comparison"],
            
            # Correlation analysis
            "correlation": ["correlation", "relationship", "associations", "correlate", "related", "relations", 
                           "dependencies", "dependency", "connections", "connected", "linked"],
            
            # Feature pairplot
            "pairplot": ["pairplot", "pair plot", "feature pairs", "variable pairs", "scatter matrix"],
            
            # Missing values
            "missing_values": ["missing values", "missing data", "empty", "null", "nan", "na", "gaps"],
            
            # Missing value handling
            "drop": ["drop", "remove", "delete", "omit", "exclude", "filter out"],
            "impute": ["impute", "fill", "interpolate", "replace", "substitute", "fill in", "complete"],
            
            # Add data
            "add_data": ["add data", "add row", "add record", "add entry", "insert data", "insert row", 
                        "insert record", "insert entry", "new data", "new row", "new record", "new entry"]
        }
        
        # Generate word synonyms for each core intent word
        self.expanded_intent_map = {}
        for intent, phrases in self.intent_map.items():
            expanded_phrases = set(phrases)
            
            for phrase in phrases:
                # Split phrase into words
                words = phrase.split()
                for word in words:
                    # Add synonyms for each word
                    synonyms = self.get_synonyms(word)
                    # Add original word + each synonym to expanded phrases
                    for syn in synonyms:
                        # Replace the word with its synonym in the phrase
                        for i, w in enumerate(words):
                            if w == word:
                                new_phrase = " ".join(words[:i] + [syn] + words[i+1:])
                                expanded_phrases.add(new_phrase)
            
            self.expanded_intent_map[intent] = list(expanded_phrases)

    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = set([word])  # Include the original word
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                synonyms.add(synonym)
                
        return list(synonyms)

    def detect_intent(self, query_lower):
        """Detect intents in the query using the expanded intent map"""
        detected_intents = set()
        
        for intent, phrases in self.expanded_intent_map.items():
            for phrase in phrases:
                if phrase.lower() in query_lower:
                    detected_intents.add(intent)
                    break
        
        return detected_intents

    def prepare_context(self, df):
        """Prepare comprehensive dataset context"""
        try:
            context_parts = []
            
            # Basic dataset information
            context_parts.append(f"Dataset Information:")
            context_parts.append(f"- Rows: {len(df)}")
            context_parts.append(f"- Columns: {len(df.columns)}")
            context_parts.append(f"- Column names: {', '.join(df.columns)}")
            
            # Data types
            context_parts.append("\nData Types:")
            for col, dtype in df.dtypes.items():
                context_parts.append(f"- {col}: {dtype}")
            
            # Numeric columns analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                context_parts.append("\nNumeric Columns Summary:")
                for col in numeric_cols:
                    desc = df[col].describe()
                    context_parts.extend([
                        f"\n{col}:",
                        f"- Mean: {desc['mean']:.2f}",
                        f"- Median: {df[col].median():.2f}",
                        f"- Std Dev: {desc['std']:.2f}",
                        f"- Min: {desc['min']:.2f}",
                        f"- Max: {desc['max']:.2f}",
                        f"- Missing values: {df[col].isnull().sum()}"
                    ])
            
            # Categorical columns analysis
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                context_parts.append("\nCategorical Columns Summary:")
                for col in cat_cols:
                    value_counts = df[col].value_counts()
                    unique_count = len(value_counts)
                    context_parts.extend([
                        f"\n{col}:",
                        f"- Unique values: {unique_count}",
                        f"- Top 3 values: {', '.join(value_counts.head(3).index)}",
                        f"- Missing values: {df[col].isnull().sum()}"
                    ])
            
            return "\n".join(context_parts)
        except Exception as e:
            return f"Error preparing context: {str(e)}"
        
    def generate_response(self, query, context):
        """Generate enhanced response using Flan-T5"""
        try:
            # Prepare prompt
            prompt = f"<|system|>You are a helpful data analysis assistant. Analyze the following dataset context and answer the question.\n\nContext: {context}\n\nQuestion: {query}</s><|user|>Please provide a detailed analysis.</s><|assistant|>"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response with safer parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("<|assistant|>")[-1].strip()
            
            # Ensure we don't return empty response
            if not assistant_response:
                return "I couldn't generate a specific analysis for your query. Please try rephrasing your question or check the data visualization tabs for insights."
                
            return assistant_response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Unable to generate response due to a technical issue. Please use the data visualization tabs for analysis."

    def extract_data_from_query(self, query, df):
        """Extract new data to be added from the query"""
        # Check if the query contains "add" or similar intent
        query_lower = query.lower()
        intents = self.detect_intent(query_lower)
        
        if "add_data" not in intents:
            return df, False, None
            
        try:
            # Extract JSON-like structures or key-value pairs from the query
            # First try to find JSON pattern {...}
            json_pattern = r'\{.*?\}'
            json_matches = re.findall(json_pattern, query)
            
            if json_matches:
                # Try to parse as JSON
                for json_str in json_matches:
                    try:
                        new_row_data = json.loads(json_str)
                        # If successful, create new row
                        return self.add_data_row(df, new_row_data)
                    except json.JSONDecodeError:
                        continue
            
            # If JSON not found or failed to parse, try key-value pattern
            # Look for patterns like "column1=value1, column2=value2" or "column1:value1, column2:value2"
            kv_pattern = r'(\w+)\s*[:=]\s*([^,]+)'
            kv_matches = re.findall(kv_pattern, query)
            
            if kv_matches:
                new_row_data = {}
                for col, val in kv_matches:
                    # Try to convert to appropriate type
                    try:
                        # Try as number first
                        if '.' in val.strip():
                            new_row_data[col] = float(val.strip())
                        else:
                            new_row_data[col] = int(val.strip())
                    except ValueError:
                        # Use as string if not a number
                        new_row_data[col] = val.strip()
                
                if new_row_data:
                    return self.add_data_row(df, new_row_data)
            
            # If structured data not found, try to find column names and values in natural language
            # This is more challenging and error-prone
            col_val_pattern = r'(?:add|set|with)?\s+(\w+)\s+(?:as|to|of|is|=|:)\s+([^,]+)'
            col_val_matches = re.findall(col_val_pattern, query_lower)
            
            if col_val_matches:
                new_row_data = {}
                for col, val in col_val_matches:
                    if col in df.columns:
                        # Try to convert val to appropriate type
                        try:
                            if '.' in val.strip():
                                new_row_data[col] = float(val.strip())
                            else:
                                new_row_data[col] = int(val.strip())
                        except ValueError:
                            new_row_data[col] = val.strip()
                
                if new_row_data:
                    return self.add_data_row(df, new_row_data)
            
            return df, False, "Could not extract data from query. Please provide data in a structured format like 'column1=value1, column2=value2'."
        
        except Exception as e:
            return df, False, f"Error extracting data from query: {str(e)}"

    def add_data_row(self, df, new_row_data):
        """Add a new row to the dataframe with averaged values for missing columns"""
        try:
            # Create a new row with default values (column averages for numeric, mode for categorical)
            new_row = {}
            
            # Calculate default values for each column
            for col in df.columns:
                if df[col].dtype.kind in 'ifc':  # integer, float, complex
                    new_row[col] = df[col].mean()
                else:
                    # For categorical or other types, use mode (most common value)
                    if not df[col].empty:
                        new_row[col] = df[col].mode()[0]
                    else:
                        new_row[col] = None
            
            # Override defaults with provided values
            for col, val in new_row_data.items():
                if col in df.columns:
                    # Try to convert value to the right type
                    try:
                        if df[col].dtype.kind in 'ifc':  # numeric
                            if isinstance(val, str):
                                # Convert string to numeric
                                try:
                                    if '.' in val:
                                        val = float(val)
                                    else:
                                        val = int(val)
                                except ValueError:
                                    # If can't convert to numeric, use mean
                                    val = df[col].mean()
                        new_row[col] = val
                    except:
                        # If conversion fails, keep the default
                        pass
                else:
                    # Skip columns that don't exist in the dataframe
                    pass
            
            # Convert to DataFrame and append to original
            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([df, new_row_df], ignore_index=True)
            
            return updated_df, True, "New data row added successfully"
        
        except Exception as e:
            return df, False, f"Error adding data row: {str(e)}"

    def perform_basic_eda(self, df, query):
        """Perform basic exploratory data analysis"""
        try:
            query_lower = query.lower()
            intents = self.detect_intent(query_lower)
            
            # Basic statistical features
            if "basic_statistics" in intents:
                st.subheader("Basic Statistical Features")
                numeric_df = df.select_dtypes(include=[np.number])
                st.dataframe(numeric_df.describe())
                
                # Additional statistics
                skewness = numeric_df.skew()
                kurtosis = numeric_df.kurt()
                st.write("Additional Statistics:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Skewness:")
                    st.dataframe(skewness)
                with col2:
                    st.write("Kurtosis:")
                    st.dataframe(kurtosis)

            # Distribution analysis
            if "distribution" in intents:
                st.subheader("Distribution Analysis")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                selected_col = st.selectbox("Select column for histogram:", numeric_cols)
                
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[selected_col],
                    nbinsx=30,
                    name="Distribution"
                ))
                fig.update_layout(
                    title=f"Distribution of {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Count",
                    template="plotly_white"
                )
                st.plotly_chart(fig)

            # Boxplot analysis
            if "boxplot" in intents:
                st.subheader("Boxplot Analysis")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                selected_col = st.selectbox("Select column for boxplot:", numeric_cols)
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=df[selected_col],
                    name=selected_col
                ))
                fig.update_layout(
                    title=f"Boxplot of {selected_col}",
                    template="plotly_white"
                )
                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in basic EDA: {str(e)}")

    def detect_and_convert_datetime_columns(self, df):
        """Auto-detect and convert potential datetime columns"""
        datetime_cols = []
        
        # First check existing datetime columns
        existing_datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(existing_datetime_cols) > 0:
            return df, list(existing_datetime_cols)
            
        # Common datetime column names to prioritize
        common_datetime_names = ['time', 'date', 'timestamp', 'datetime', 'created_at', 
                                'updated_at', 'time_stamp', 'date_time', 'day']
        
        # First try columns with common names
        potential_cols = []
        for col in df.columns:
            col_lower = col.lower()
            if any(dt_name in col_lower for dt_name in common_datetime_names):
                potential_cols.append(col)
        
        # Then add other string columns as potential datetime columns
        for col in df.select_dtypes(include=['object']).columns:
            if col not in potential_cols:
                # Check first few values for date patterns
                sample = df[col].dropna().head(5).astype(str)
                if sample.empty:
                    continue
                    
                # Look for basic date markers (slashes, dashes, colons)
                date_markers = ['/', '-', ':', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                has_date_marker = any(any(marker in str(val).lower() for marker in date_markers) for val in sample)
                
                if has_date_marker:
                    potential_cols.append(col)
        
        # Try to convert each potential column
        for col in potential_cols:
            try:
                # Try multiple datetime parsing formats
                formats_to_try = [
                    # Try pandas auto-detection first
                    None,  
                    # Then try common formats
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%d-%m-%Y',
                    '%d/%m/%Y',
                    '%m/%d/%Y',
                    '%d-%m-%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S',
                    '%m/%d/%Y %H:%M:%S',
                    '%Y/%m/%d',
                    '%Y/%m/%d %H:%M:%S',
                    '%d-%b-%Y',
                    '%d %b %Y',
                    '%b %d %Y',
                    '%d.%m.%Y',
                    '%m.%d.%Y',
                    '%Y.%m.%d',
                ]
                
                for fmt in formats_to_try:
                    try:
                        if fmt is None:
                            # Try pandas automatic parsing
                            df[col] = pd.to_datetime(df[col], errors='raise')
                        else:
                            # Try with specific format
                            df[col] = pd.to_datetime(df[col], format=fmt, errors='raise')
                        
                        # If successful, add to the list of datetime columns
                        datetime_cols.append(col)
                        break
                    except (ValueError, TypeError):
                        continue
            except:
                continue
        
        return df, datetime_cols

    def perform_time_series_analysis(self, df, query):
        """Perform time series analysis with improved datetime detection"""
        try:
            query_lower = query.lower()
            intents = self.detect_intent(query_lower)
            
            time_related_intents = {"time_series", "hourly", "daily", "monthly", "yearly", 
                                  "weekday", "weekend", "compare_halves"}
            
            if any(intent in intents for intent in time_related_intents):
                
                # First detect and convert potential datetime columns
                df, datetime_cols = self.detect_and_convert_datetime_columns(df)
                
                if not datetime_cols:
                    st.warning("No datetime column could be detected. Please ensure your dataset contains properly formatted date/time information.")
                    return
                
                # Let user choose datetime column if multiple were detected
                if len(datetime_cols) > 1:
                    selected_date_col = st.selectbox(
                        "Multiple time columns detected. Select the time column to use:",
                        datetime_cols
                    )
                    date_col = selected_date_col
                else:
                    date_col = datetime_cols[0]
                    st.info(f"Using {date_col} as the time column for analysis.")
                
                # Get numeric columns for analysis
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    st.warning("No numeric columns found for time series analysis.")
                    return
                    
                # Basic time series visualization
                st.subheader("Time Series Visualization")
                selected_numeric_cols = st.multiselect(
                    "Select numeric columns to visualize:",
                    numeric_cols,
                    default=[numeric_cols[0]] if numeric_cols else []
                )
                
                if selected_numeric_cols:
                    fig = go.Figure()
                    for col in selected_numeric_cols:
                        fig.add_trace(go.Scatter(
                            x=df[date_col],
                            y=df[col],
                            name=col
                        ))
                    
                    fig.update_layout(
                        title="Time Series Analysis",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig)
                    
                    # Display time range information
                    min_date = df[date_col].min()
                    max_date = df[date_col].max()
                    st.write(f"Time Range: {min_date} to {max_date}")
                    
                    # Time duration
                    duration = max_date - min_date
                    st.write(f"Time Span: {duration}")
                
                # Compare first and second half trends
                if "compare_halves" in intents:
                    st.subheader("First Half vs Second Half Comparison")
                    mid_point = df[date_col].min() + (df[date_col].max() - df[date_col].min()) / 2
                    
                    first_half = df[df[date_col] <= mid_point]
                    second_half = df[df[date_col] > mid_point]
                    
                    # Show period information
                    st.write(f"First Half Period: {first_half[date_col].min()} to {first_half[date_col].max()}")
                    st.write(f"Second Half Period: {second_half[date_col].min()} to {second_half[date_col].max()}")
                    
                    selected_comparison_col = st.selectbox(
                        "Select column to compare:", 
                        numeric_cols
                    )
                    
                    fig = go.Figure()
                    
                    # Add first half data
                    fig.add_trace(go.Scatter(
                        x=first_half[date_col],
                        y=first_half[selected_comparison_col],
                        name=f"{selected_comparison_col} (First Half)",
                        line=dict(color='blue')
                    ))
                    
                    # Add second half data
                    fig.add_trace(go.Scatter(
                        x=second_half[date_col],
                        y=second_half[selected_comparison_col],
                        name=f"{selected_comparison_col} (Second Half)",
                        line=dict(color='red')
                    ))
                    
                    fig.update_layout(
                        title="First Half vs Second Half Comparison",
                        xaxis_title="Date",
                        yaxis_title=selected_comparison_col,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig)
                    
                    # Statistical comparison
                    comparison_data = {
                        "Metric": ["Mean", "Median", "Std Dev", "Min", "Max"],
                        "First Half": [
                            first_half[selected_comparison_col].mean(),
                            first_half[selected_comparison_col].median(),
                            first_half[selected_comparison_col].std(),
                            first_half[selected_comparison_col].min(),
                            first_half[selected_comparison_col].max()
                        ],
                        "Second Half": [
                            second_half[selected_comparison_col].mean(),
                            second_half[selected_comparison_col].median(),
                            second_half[selected_comparison_col].std(),
                            second_half[selected_comparison_col].min(),
                            second_half[selected_comparison_col].max()
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.write("Statistical Comparison:")
                    st.dataframe(comparison_df)

                # Hourly trend analysis
                if "hourly" in intents:
                    st.subheader("Hourly Trend Analysis")
                    
                    # Extract hour information
                    try:
                        df['hour'] = df[date_col].dt.hour
                        
                        # Get unique dates
                        df['date_only'] = df[date_col].dt.date
                        available_dates = sorted(df['date_only'].unique())
                        
                        if len(available_dates) > 0:
                            selected_date = st.selectbox(
                                "Select date for hourly analysis:", 
                                available_dates
                            )
                            
                            daily_data = df[df['date_only'] == selected_date]
                            
                            if not daily_data.empty:
                                selected_hour_col = st.selectbox(
                                    "Select column for hourly analysis:", 
                                    numeric_cols
                                )
                                
                                # Group by hour
                                hourly_data = daily_data.groupby('hour')[selected_hour_col].agg(['mean', 'min', 'max']).reset_index()
                                
                                fig = go.Figure()
                                
                                # Add mean line
                                fig.add_trace(go.Scatter(
                                    x=hourly_data['hour'],
                                    y=hourly_data['mean'],
                                    name="Mean",
                                    line=dict(color='blue')
                                ))
                                
                                # Add range
                                fig.add_trace(go.Scatter(
                                    x=hourly_data['hour'],
                                    y=hourly_data['min'],
                                    name="Min",
                                    line=dict(color='lightblue', dash='dash')
                                ))
                                
                                fig.add_trace(go.Scatter(
                                    x=hourly_data['hour'],
                                    y=hourly_data['max'],
                                    name="Max",
                                    line=dict(color='lightblue', dash='dash'),
                                    fill='tonexty'
                                ))
                                
                                fig.update_layout(
                                    title=f"Hourly Trend for {selected_hour_col} on {selected_date}",
                                    xaxis_title="Hour of Day",
                                    yaxis_title=selected_hour_col,
                                    xaxis=dict(tickmode='linear', tick0=0, dtick=1),
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig)
                        else:
                            st.warning("No daily data available for hourly analysis.")
                    except Exception as e:
                        st.error(f"Error in hourly analysis: {str(e)}")

                # Weekday vs Weekend analysis
                if "weekday" in intents and "weekend" in intents:
                    st.subheader("Weekday vs Weekend Analysis")
                    
                    # Extract day of week
                    try:
                        df['day_of_week'] = df[date_col].dt.dayofweek
                        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 5=Saturday, 6=Sunday
                        
                        weekday_data = df[~df['is_weekend']]
                        weekend_data = df[df['is_weekend']]
                        
                        st.write(f"Weekday Records: {len(weekday_data)}")
                        st.write(f"Weekend Records: {len(weekend_data)}")
                        
                        selected_comparison_col = st.selectbox(
                            "Select column for weekday/weekend comparison:", 
                            numeric_cols,
                            key="weekend_comparison"
                        )
                        
                        # Create box plots for comparison
                        fig = go.Figure()
                        
                        fig.add_trace(go.Box(
                            y=weekday_data[selected_comparison_col],
                            name="Weekday",
                            boxpoints='outliers'
                        ))
                        
                        fig.add_trace(go.Box(
                            y=weekend_data[selected_comparison_col],
                            name="Weekend",
                            boxpoints='outliers'
                        ))
                        
                        fig.update_layout(
                            title=f"Weekday vs Weekend: {selected_comparison_col}",
                            yaxis_title=selected_comparison_col,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig)
                        
                        # Statistical comparison
                        comparison_data = {
                            "Metric": ["Mean", "Median", "Std Dev", "Min", "Max", "Count"],
                            "Weekday": [
                                weekday_data[selected_comparison_col].mean(),
                                weekday_data[selected_comparison_col].median(),
                                weekday_data[selected_comparison_col].std(),
                                weekday_data[selected_comparison_col].min(),
                                weekday_data[selected_comparison_col].max(),
                                len(weekday_data)
                            ],
                            "Weekend": [
                                weekend_data[selected_comparison_col].mean(),
                                weekend_data[selected_comparison_col].median(),
                                weekend_data[selected_comparison_col].std(),
                                weekend_data[selected_comparison_col].min(),
                                weekend_data[selected_comparison_col].max(),
                                len(weekend_data)
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.write("Statistical Comparison:")
                        st.dataframe(comparison_df)
                        
                        # Calculate percent difference
                        if not weekday_data.empty and not weekend_data.empty:
                            weekday_mean = weekday_data[selected_comparison_col].mean()
                            weekend_mean = weekend_data[selected_comparison_col].mean()
                            if weekday_mean != 0:
                                percent_diff = ((weekend_mean - weekday_mean) / weekday_mean) * 100
                                st.write(f"Weekend vs Weekday Percent Difference: {percent_diff:.2f}%")
                                
                    except Exception as e:
                        st.error(f"Error in weekday/weekend analysis: {str(e)}")
                        
                # Monthly analysis
                if "monthly" in intents or "month" in intents:
                    st.subheader("Monthly Analysis")
                    
                    try:
                        # Extract month information
                        df['month'] = df[date_col].dt.month
                        df['year'] = df[date_col].dt.year
                        df['year_month'] = df[date_col].dt.strftime('%Y-%m')
                        
                        # Get unique year-months
                        available_year_months = sorted(df['year_month'].unique())
                        
                        selected_ts_col = st.selectbox(
                            "Select column for monthly analysis:", 
                            numeric_cols,
                            key="monthly_analysis"
                        )
                        
                        # Group by year-month
                        monthly_data = df.groupby('year_month')[selected_ts_col].agg(['mean', 'min', 'max', 'count']).reset_index()
                        
                        fig = go.Figure()
                        
                        # Add mean line
                        fig.add_trace(go.Scatter(
                            x=monthly_data['year_month'],
                            y=monthly_data['mean'],
                            mode='lines+markers',
                            name="Mean",
                            line=dict(color='blue')
                        ))
                        
                        # Add range area
                        fig.add_trace(go.Scatter(
                            x=monthly_data['year_month'],
                            y=monthly_data['min'],
                            name="Min",
                            line=dict(color='lightblue', dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=monthly_data['year_month'],
                            y=monthly_data['max'],
                            name="Max",
                            line=dict(color='lightblue', dash='dash'),
                            fill='tonexty'
                        ))
                        
                        fig.update_layout(
                            title=f"Monthly Trend for {selected_ts_col}",
                            xaxis_title="Month",
                            yaxis_title=selected_ts_col,
                            template="plotly_white"
                        )
                        st.plotly_chart(fig)
                        
                        # Sample size visualization
                        fig2 = go.Figure()
                        fig2.add_trace(go.Bar(
                            x=monthly_data['year_month'],
                            y=monthly_data['count'],
                            name="Record Count"
                        ))
                        
                        fig2.update_layout(
                            title="Records per Month",
                            xaxis_title="Month",
                            yaxis_title="Number of Records",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig2)
                        
                    except Exception as e:
                        st.error(f"Error in monthly analysis: {str(e)}")

        except Exception as e:
                st.error(f"Error in time series analysis: {str(e)}")
                st.exception(e)

    def perform_correlation_analysis(self, df, query):
        """Perform correlation analysis"""
        try:
            query_lower = query.lower()
            
            if "correlation" in query_lower:
                st.subheader("Correlation Analysis")
                
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                
                # Correlation heatmap
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix Heatmap",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig)
                
                # Top 5 correlations
                if "top 5" in query_lower:
                    st.write("Top 5 Correlations:")
                    correlations = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i):
                            correlations.append({
                                'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                                'Correlation': abs(corr_matrix.iloc[i, j])
                            })
                    
                    correlations_df = pd.DataFrame(correlations)
                    st.dataframe(correlations_df.nlargest(5, 'Correlation'))

            # Pairplot
            if "pairplot" in query_lower:
                st.subheader("Feature Pairplot")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                selected_cols = st.multiselect(
                    "Select features for pairplot:",
                    numeric_cols,
                    default=list(numeric_cols)[:4]
                )
                
                if selected_cols:
                    fig = px.scatter_matrix(
                        df[selected_cols],
                        dimensions=selected_cols,
                        title="Feature Pairplot"
                    )
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")

    def handle_missing_values(self, df, query):
        """Handle missing values analysis"""
        try:
            query_lower = query.lower()
            
            if "missing values" in query_lower:
                st.subheader("Missing Values Analysis")
                
                # Calculate missing values
                missing_vals = df.isnull().sum()
                missing_percentages = (missing_vals / len(df) * 100).round(2)

                # Create missing values bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=missing_percentages.index,
                    y=missing_percentages.values,
                    name="Missing Percentage"
                ))
                fig.update_layout(
                    title="Percentage of Missing Values by Column",
                    xaxis_title="Column",
                    yaxis_title="Missing Percentage",
                    template="plotly_white"
                )
                st.plotly_chart(fig)
                
                # Handle missing values
                if "drop" in query_lower:
                    st.write(f"Original shape: {df.shape}")
                    cleaned_df = df.dropna()
                    st.write(f"Shape after dropping missing values: {cleaned_df.shape}")
                    return cleaned_df
                
                elif "impute" in query_lower:
                    st.write("Before imputation:")
                    st.write(df.isnull().sum())
                    
                    imputed_df = df.interpolate(method='linear')
                    
                    st.write("After imputation:")
                    st.write(imputed_df.isnull().sum())
                    return imputed_df
                
                return df

        except Exception as e:
            st.error(f"Error handling missing values: {str(e)}")
            return df

    def run(self):
        """Main application logic"""
        try:
            st.sidebar.info("""
            📊 This application uses Flan-T5 for advanced data analysis.
            Upload your dataset or use the sample data to get started.
            """)

            # Data source selection
            data_source = st.sidebar.radio(
                "Select Data Source:",
                ["Use Sample Data", "Upload Your Own Data"]
            )

            df = None

            # Load data
            if data_source == "Upload Your Own Data":
                uploaded_file = st.sidebar.file_uploader(
                    "Upload your dataset (CSV, Excel, or JSON)",
                    type=['csv', 'xlsx', 'xls', 'json']
                )
                if uploaded_file:
                    try:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        if file_ext == 'csv':
                            df = pd.read_csv(uploaded_file)
                        elif file_ext in ['xls', 'xlsx']:
                            df = pd.read_excel(uploaded_file)
                        elif file_ext == 'json':
                            df = pd.read_json(uploaded_file)
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        df = None
            else:
                # Load sample data with timestamp
                try:
                    dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
                    df = pd.DataFrame({
                        'Timestamp': dates,
                        'Efficiency': np.random.normal(80, 5, 100),
                        'Steam_Temperature': np.random.normal(150, 10, 100),
                        'Steam_Pressure': np.random.normal(50, 5, 100),
                        'Fuel_Flow': np.random.normal(100, 15, 100),
                        'Steam_Flow': np.random.normal(75, 10, 100)
                    })
                except Exception as e:
                    st.error(f"Error creating sample data: {str(e)}")

            # Check if df is None or empty
            if df is None or df.empty:
                st.warning("Please upload a dataset or use the sample data to begin analysis.")
                return

            # Continue with the analysis only if we have valid data
            st.sidebar.subheader("Dataset Preview")
            st.sidebar.dataframe(df.head(3))

            # Query interface
            st.subheader("Ask Questions About Your Data")
            query = st.text_input(
                "Enter your question:",
                placeholder="e.g., Give me basic statistical features for the data"
            )
            if query:
                try:
                    with st.spinner("Analyzing your data..."):
                        # Perform analyses based on query
                        self.perform_basic_eda(df, query)
                        self.perform_time_series_analysis(df, query)
                        self.perform_correlation_analysis(df, query)
                        df = self.handle_missing_values(df, query)
                        
                        # Extract and add new data from query if applicable
                        df, success, message = self.extract_data_from_query(query, df)
                        if success:
                            st.success(message)
                        
                        # Generate response using Flan-T5
                        context = self.prepare_context(df)
                        response = self.generate_response(query, context)
                        
                        st.markdown("### AI Assistant Response")
                        st.write(response)
                        
                        # Only add download button if we got a valid response
                        if response and response != "Unable to generate response due to a technical issue. Please use the data visualization tabs for analysis.":
                            analysis_text = f"""
                            Analysis Report
                            --------------
                            Query: {query}
                            
                            Model Response:
                            {response}
                            
                            Dataset Shape: {df.shape}
                            Columns: {', '.join(df.columns)}
                            """
                            st.download_button(
                                label="Download Analysis Report",
                                data=analysis_text,
                                file_name="analysis_report.txt",
                                mime="text/plain"
                            )

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")
                    st.info("You can still use the visualization tools below to explore your data.")
                            
                # Add enhanced data exploration tools
            with st.expander("Advanced Data Exploration Tools"):
                    tab1, tab2, tab3, tab4 = st.tabs(["Statistical Analysis", "Time Series", "Correlations", "Missing Values"])
                    
                    with tab1:
                        st.subheader("Statistical Analysis")
                        
                        # Column selection for detailed analysis
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            selected_column = st.selectbox("Select a column for detailed analysis:", numeric_cols)
                        
                            if selected_column:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Basic statistics
                                    stats = df[selected_column].describe()
                                    st.write("Basic Statistics:")
                                    st.dataframe(stats)
                                    
                                    # Additional metrics
                                    st.write("Additional Metrics:")
                                    metrics = {
                                        "Skewness": df[selected_column].skew(),
                                        "Kurtosis": df[selected_column].kurtosis(),
                                        "Missing Values": df[selected_column].isnull().sum(),
                                        "Missing Percentage": (df[selected_column].isnull().sum() / len(df) * 100).round(2)
                                    }
                                    st.dataframe(pd.Series(metrics))
                                
                                with col2:
                                    # Distribution plot
                                    fig = go.Figure()
                                    fig.add_trace(go.Histogram(
                                        x=df[selected_column],
                                        nbinsx=30,
                                        name="Distribution"
                                    ))
                                    fig.add_trace(go.Box(
                                        y=df[selected_column],
                                        name="Box Plot",
                                        boxpoints="outliers"
                                    ))
                                    fig.update_layout(
                                        title=f"Distribution and Box Plot of {selected_column}",
                                        template="plotly_white"
                                    )
                                    st.plotly_chart(fig)
                        
                        else:
                            st.warning("No numeric columns found in the dataset.")
                    
                    with tab2:
                        st.subheader("Time Series Analysis")
                        
                        # Identify datetime columns
                        datetime_cols = df.select_dtypes(include=['datetime64']).columns
                        if len(datetime_cols) == 0:
                            st.warning("No datetime columns detected. Please ensure your data includes a timestamp column.")
                        else:
                            time_col = st.selectbox("Select timestamp column:", datetime_cols)
                            value_col = st.selectbox("Select value column:", numeric_cols)
                            
                            # Time series decomposition
                            try:
                                df_temp = df.set_index(time_col)
                                df_temp = df_temp[value_col].resample('H').mean()
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=df_temp.index,
                                    y=df_temp.values,
                                    name="Original"
                                ))
                                
                                # Add trend line
                                z = np.polyfit(range(len(df_temp)), df_temp.values, 1)
                                p = np.poly1d(z)
                                fig.add_trace(go.Scatter(
                                    x=df_temp.index,
                                    y=p(range(len(df_temp))),
                                    name="Trend",
                                    line=dict(dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title=f"Time Series Analysis of {value_col}",
                                    xaxis_title="Time",
                                    yaxis_title=value_col,
                                    template="plotly_white"
                                )
                                st.plotly_chart(fig)
                                
                            except Exception as e:
                                st.error(f"Error in time series analysis: {str(e)}")
                    
                    with tab3:
                        st.subheader("Correlation Analysis")
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) >= 2:
                        
                        # Correlation matrix
                            corr_matrix = df.select_dtypes(include=[np.number]).corr()
                            
                            # Heatmap
                            fig = px.imshow(
                                corr_matrix,
                                title="Correlation Matrix Heatmap",
                                color_continuous_scale="RdBu",
                                aspect="auto"
                            )
                            st.plotly_chart(fig)
                            
                            # Top correlations
                            st.write("Top 5 Correlations:")
                            correlations = []
                            for i in range(len(corr_matrix.columns)):
                                for j in range(i):
                                    correlations.append({
                                        'Variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                                        'Correlation': abs(corr_matrix.iloc[i, j])
                                    })
                            
                            correlations_df = pd.DataFrame(correlations)
                            st.dataframe(correlations_df.nlargest(5, 'Correlation'))

                        else:
                            st.warning("At least two numeric columns are required for correlation analysis.")

                    with tab4:
                        st.subheader("Missing Values Analysis")
                        
                        # Missing values summary
                        missing_summary = pd.DataFrame({
                            'Missing Values': df.isnull().sum(),
                            'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                        })
                        
                        # Display summary
                        st.write("Missing Values Summary:")
                        st.dataframe(missing_summary)
                        
                        # Missing values visualization
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=missing_summary.index,
                            y=missing_summary['Percentage'],
                            name="Missing Percentage"
                        ))
                        fig.update_layout(
                            title="Percentage of Missing Values by Column",
                            xaxis_title="Column",
                            yaxis_title="Missing Percentage (%)",
                            template="plotly_white"
                        )
                        st.plotly_chart(fig)
                        
                        # Missing values handling options
                        handling_method = st.selectbox(
                            "Select missing values handling method:",
                            ["None", "Drop rows", "Linear interpolation", "Forward fill", "Backward fill"]
                        )
                        
                        if handling_method != "None":
                            if handling_method == "Drop rows":
                                cleaned_df = df.dropna()
                            elif handling_method == "Linear interpolation":
                                cleaned_df = df.interpolate(method='linear')
                            elif handling_method == "Forward fill":
                                cleaned_df = df.fillna(method='ffill')
                            else:  # Backward fill
                                cleaned_df = df.fillna(method='bfill')
                            
                            st.write(f"Original shape: {df.shape}")
                            st.write(f"Shape after handling missing values: {cleaned_df.shape}")
                            
                            # Update the main dataframe if user confirms
                            if st.button("Apply missing values handling"):
                                df = cleaned_df
                                st.success("Missing values handled successfully!")

        except Exception as e:
                st.error(f"Error in application: {str(e)}")

if __name__ == "__main__":
    # Set PyTorch configurations
    torch.set_default_dtype(torch.float32)
    
    # Run application
    app = DatasetQueryAssistant()
    app.run()