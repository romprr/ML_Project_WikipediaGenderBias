import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os
import glob
import urllib.parse

st.set_page_config(layout="wide", page_title="CS Wikipedia Analysis", page_icon="")
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    h1, h2, h3 { color: #00d2ff; font-family: 'Inter', sans-serif; }
    .stPlotlyChart { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
</style>
""", unsafe_allow_html=True)

st.title(" Computer Science Wikipedia: Gender & Citation Analysis")

GENDER_COLORS = {"Male": "#1f77b4", "Female": "#FFBCAE", "Unknown": "gray", "male": "#1f77b4", "female": "#FFBCAE", "unknown": "gray"}

@st.cache_data
def load_metadata():
    meta_path = "output/viz_metadata_patched.parquet" if os.path.exists("output/viz_metadata_patched.parquet") else "output/viz_metadata.parquet"
    if os.path.exists(meta_path):
        return pl.read_parquet(meta_path)
    return pl.DataFrame()

@st.cache_data
def load_cs_topics():
    cs_topics = set()
    files = glob.glob("data/wikidata_cs_parquet_batches/*.parquet")
    for f in files:
        df_cs = pd.read_parquet(f)
        if 'enwiki' in df_cs.columns:
            for url in df_cs['enwiki'].dropna():
                title = urllib.parse.unquote(str(url).split("/wiki/")[-1]).replace("_", " ")
                cs_topics.add(title)
    return cs_topics

@st.cache_data
def load_sentiment():
    if os.path.exists("output/top200_ml_sentiment_analysis.parquet"):
        return pl.read_parquet("output/top200_ml_sentiment_analysis.parquet")
    return pl.DataFrame()

@st.cache_data
def get_total_wikipedia_pages():
    try:
        if not os.path.exists("data/translation_table.csv"):
            return 0
        # Fast binary buffered line counting (cross-platform compatible)
        with open("data/translation_table.csv", "rb") as f:
            return sum(1 for _ in f) - 1  # -1 for the header
    except Exception:
        return 0

def get_cites_lazy():
    if os.path.exists("output/viz_phase3_citations.parquet"):
        return pl.scan_parquet("output/viz_phase3_citations.parquet")
    return None

df_meta = load_metadata()
cs_topics = load_cs_topics()

if df_meta.is_empty():
    st.error("Metadata not found. Please run the data pipelines.")
    st.stop()

# --- FILTERING ---
# Compute the subset of people directly tied to CS to avoid loading everything
cs_occupations = ["computer scientist", "programmer", "software engineer", "mathematician", "engineer", "information scientist", "data scientist", "web developer"]

lazy_cites = get_cites_lazy()
if lazy_cites is not None:
    # Compute this once lazily
    cs_cited = lazy_cites.filter(pl.col("source_page").is_in(list(cs_topics))).select("cited_person").unique().collect(engine="streaming")["cited_person"].to_list()
else:
    cs_cited = []

df_meta = df_meta.with_columns(
    pl.col("person_title").is_in(cs_cited).alias("cited_in_cs_topic"),
    pl.col("occupation").str.to_lowercase().is_in(cs_occupations).alias("has_cs_job")
)
df_cs_people = df_meta.filter(pl.col("cited_in_cs_topic") | pl.col("has_cs_job"))

page = st.sidebar.radio("Navigation", [
    "Data Exploration",
    "Section Analytics",
    "Citation Analytics",
    "Sentiment & Rank Analysis",
    "Galaxy Network Plots",
    "Language Bias (Phase 7)"
])

if page == "Data Exploration":
    current_year = 2024
    
    st.header("Global Wikipedia Demographics (All Domains)")
    global_total = len(df_meta)
    global_men = len(df_meta.filter(pl.col("gender") == "male"))
    global_women = len(df_meta.filter(pl.col("gender") == "female"))
    global_unknown_gender = len(df_meta.filter(~pl.col("gender").is_in(["male", "female"])))
    
    global_missing_birth = len(df_meta.filter(pl.col("birth_year").is_null()))
    global_alive = len(df_meta.filter(pl.col("death_year").is_null()))
    global_missing_death = len(df_meta.filter(
        (pl.col("birth_year").is_not_null()) & 
        (pl.col("death_year").is_null()) & 
        ((current_year - pl.col("birth_year")) > 110)
    ))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Valid Pages Extracted (Bios + CS)", f"{get_total_wikipedia_pages():,}")
    col2.metric("Total Humans (Wikidata)", f"{global_total:,}")
    
    df_cites_lazy = get_cites_lazy()
    total_cites = df_cites_lazy.select(pl.len()).collect().item() if df_cites_lazy is not None else 0
    col3.metric("Total Citations Extracted", f"{total_cites:,}")
    
    col4.metric("CS Topics Identified", f"{len(cs_topics):,}")
    
    occ_files = glob.glob("data/wikidata_occupations_parquet_batches/*.parquet")
    total_occs = sum(pd.read_parquet(f).shape[0] for f in occ_files) if occ_files else 0
    col5.metric("Wikidata Occupations Loaded", f"{total_occs:,}")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    c1.metric("Men", f"{global_men:,}")
    c2.metric("Women", f"{global_women:,}")
    
    col4, col5, col6, col7 = st.columns(4)
    col4.metric("Missing Birth Dates", f"{global_missing_birth:,}")
    col5.metric("Missing Death Dates (Total)", f"{global_alive:,}")
    col6.metric("Missing Death (>110yo)", f"{global_missing_death:,}")
    col7.metric("Unknown Gender", f"{global_unknown_gender:,}")
    
    # Global Gender pie chart
    global_gender_counts = pd.DataFrame({
        "Gender": ["Male", "Female", "Unknown"],
        "Count": [global_men, global_women, global_total - global_men - global_women]
    })
    fig_global_gender = px.pie(global_gender_counts, values="Count", names="Gender", title="Global Gender Distribution", color="Gender", color_discrete_map=GENDER_COLORS)
    st.plotly_chart(fig_global_gender, use_container_width=True)
    
    st.subheader("Global Birth Year Distribution by Gender")
    global_birth_counts = df_meta.filter(
        pl.col("gender").is_in(["male", "female"]) & 
        (pl.col("birth_year") > 0)
    ).group_by(["birth_year", "gender"]).len().sort("birth_year").to_pandas()
    fig_global_age = px.line(global_birth_counts, x="birth_year", y="len", color="gender", title="Global Birth Year Distribution (Men vs Women)", color_discrete_map=GENDER_COLORS)
    st.plotly_chart(fig_global_age, use_container_width=True)

    st.markdown("---")
    st.header("Computer Science Demographics")
    
    cs_total = len(df_cs_people)
    cs_men = len(df_cs_people.filter(pl.col("gender") == "male"))
    cs_women = len(df_cs_people.filter(pl.col("gender") == "female"))
    cs_unknown_gender = len(df_cs_people.filter(~pl.col("gender").is_in(["male", "female"])))
    
    cs_missing_birth = len(df_cs_people.filter(pl.col("birth_year").is_null()))
    cs_alive = len(df_cs_people.filter(pl.col("death_year").is_null()))
    cs_missing_death = len(df_cs_people.filter(
        (pl.col("birth_year").is_not_null()) & 
        (pl.col("death_year").is_null()) & 
        ((current_year - pl.col("birth_year")) > 110)
    ))
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total CS Profiles", f"{cs_total:,}")
    c2.metric("CS Men", f"{cs_men:,}")
    c3.metric("CS Women", f"{cs_women:,}")
    
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Missing Birth Dates", f"{cs_missing_birth:,}")
    c5.metric("Missing Death Dates (Total)", f"{cs_alive:,}")
    c6.metric("Missing Death (>110yo)", f"{cs_missing_death:,}")
    c7.metric("Unknown Gender", f"{cs_unknown_gender:,}")
    
    gender_counts = pd.DataFrame({
        "Gender": ["Male", "Female", "Unknown"],
        "Count": [cs_men, cs_women, cs_total - cs_men - cs_women]
    })
    fig_gender = px.pie(gender_counts, values="Count", names="Gender", title="CS Gender Distribution", color="Gender", color_discrete_map=GENDER_COLORS)
    st.plotly_chart(fig_gender, use_container_width=True)
    
    st.subheader("CS Birth Year Distribution by Gender")
    pure_cs_regex = "(?i)(computer scientist|programmer|software engineer|information scientist|data scientist|web developer)"
    cs_birth_counts = df_cs_people.filter(
        pl.col("gender").is_in(["male", "female"]) & 
        (pl.col("birth_year") > 0) &
        pl.col("occupation").str.contains(pure_cs_regex)
    ).group_by(["birth_year", "gender"]).len().sort("birth_year").to_pandas()
    fig_cs_age = px.line(cs_birth_counts, x="birth_year", y="len", color="gender", title="CS Birth Year Distribution (Men vs Women, Pure CS Occupations Only)", color_discrete_map=GENDER_COLORS)
    st.plotly_chart(fig_cs_age, use_container_width=True)

    st.subheader("Top Occupations in CS")
    occs = df_cs_people.select("occupation").drop_nulls().to_pandas()
    all_occs = occs["occupation"].str.split('|').explode().value_counts().head(20).reset_index()
    all_occs.columns = ["Occupation", "Count"]
    fig_occ = px.bar(all_occs, x="Occupation", y="Count", title="Overall Top 20 Occupations in CS")
    st.plotly_chart(fig_occ, use_container_width=True)
    
    st.subheader("Top Occupations by Gender")
    colA, colB = st.columns(2)
    
    with colA:
        occs_men = df_cs_people.filter(pl.col("gender") == "male").select("occupation").drop_nulls().to_pandas()
        all_occs_men = occs_men["occupation"].str.split('|').explode().value_counts().head(15).reset_index()
        all_occs_men.columns = ["Occupation", "Count"]
        fig_occ_men = px.bar(all_occs_men, x="Occupation", y="Count", title="Top 15 Occupations (Men)", color_discrete_sequence=[GENDER_COLORS["Male"]])
        st.plotly_chart(fig_occ_men, use_container_width=True)
        
    with colB:
        occs_women = df_cs_people.filter(pl.col("gender") == "female").select("occupation").drop_nulls().to_pandas()
        all_occs_women = occs_women["occupation"].str.split('|').explode().value_counts().head(15).reset_index()
        all_occs_women.columns = ["Occupation", "Count"]
        fig_occ_women = px.bar(all_occs_women, x="Occupation", y="Count", title="Top 15 Occupations (Women)", color_discrete_sequence=[GENDER_COLORS["Female"]])
        st.plotly_chart(fig_occ_women, use_container_width=True)

elif page == "Section Analytics":
    st.header("Section Analytics")
    p2_files = glob.glob("data/phase2_parsed_data_part_*.parquet")
    if p2_files:
        # Load lazily and stream the join
        lazy_sec = pl.scan_parquet(p2_files).select(["page_title", "classification", "words", "sentences", "letters"])
        df_sec_merged = lazy_sec.join(df_cs_people.lazy().select(["person_title", "gender"]), left_on="page_title", right_on="person_title", how="inner")
        
        sec_types = df_sec_merged.group_by(["gender", "classification"]).len().collect(engine="streaming").to_pandas()
        sec_types = sec_types[sec_types['gender'].isin(['male', 'female'])]
        sec_types['percentage'] = sec_types.groupby('gender')['len'].transform(lambda x: (x / x.sum()) * 100)
        
        fig_sectypes = px.bar(sec_types, x="gender", y="percentage", color="classification", 
                              title="Section Topic Ratio (%) by Gender", barmode="group",
                              text=sec_types['percentage'].apply(lambda x: f'{x:.1f}%'))
        fig_sectypes.update_traces(textposition='outside')
        fig_sectypes.update_layout(yaxis_title="Percentage (%)", yaxis_range=[0, max(sec_types['percentage']) + 15])
        st.plotly_chart(fig_sectypes, use_container_width=True)
        
        sections_per_page = df_sec_merged.group_by(["page_title", "gender"]).len()
        agg_num_sections = sections_per_page.group_by("gender").agg(
            pl.col("len").mean().round(2).alias("avg_sections_per_page"),
            pl.col("len").median().round(2).alias("median_sections_per_page")
        )

        agg_sec = df_sec_merged.group_by("gender").agg(
            pl.col("words").mean().round(1).alias("avg_words"),
            pl.col("words").median().round(1).alias("median_words"),
            pl.col("words").max().alias("max_words"),
            pl.col("words").min().alias("min_words"),
            pl.col("sentences").mean().round(1).alias("avg_paragraphs"),
            pl.col("sentences").median().round(1).alias("median_paragraphs"),
            pl.col("letters").mean().round(1).alias("avg_chars"),
            pl.col("letters").median().round(1).alias("median_chars")
        )
        
        final_agg = agg_sec.join(agg_num_sections, on="gender").collect(engine="streaming").to_pandas()
        st.write("### Section Length Metrics (Averages & Extremes)")
        st.dataframe(final_agg)

elif page == "Citation Analytics":
    st.header("Citation Analytics")
    if lazy_cites is not None:
        # Strictly filter citations to only those originating from CS Topics
        cs_cites_filtered = lazy_cites.filter(pl.col("source_page").is_in(list(cs_topics)))
        cs_cites = cs_cites_filtered.join(df_cs_people.lazy().select("person_title"), left_on="cited_person", right_on="person_title", how="inner")
        cite_counts = cs_cites.group_by(["cited_person", "gender"]).len().sort("len", descending=True).collect(engine="streaming")
        
        top_men = cite_counts.filter(pl.col("gender") == "male").head(100)
        top_women = cite_counts.filter(pl.col("gender") == "female").head(100)
        top_100_combined = pl.concat([top_men, top_women])
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Avg Citations (Top 100 Men)", f"{top_men['len'].mean():.1f}")
            st.metric("Median Citations (Top 100 Men)", f"{top_men['len'].median():.1f}")
        with col_c2:
            st.metric("Avg Citations (Top 100 Women)", f"{top_women['len'].mean():.1f}")
            st.metric("Median Citations (Top 100 Women)", f"{top_women['len'].median():.1f}")
        
        st.markdown("---")
        top_10_combined = pl.concat([top_men.head(10), top_women.head(10)]).to_pandas()
        fig_top10 = px.bar(top_10_combined, x="cited_person", y="len", color="gender", title="Top 10 Most Cited Men and Women", barmode="group", color_discrete_map=GENDER_COLORS)
        st.plotly_chart(fig_top10, use_container_width=True)
        
        cites_with_birth = cs_cites.join(df_cs_people.lazy().select(["person_title", "birth_year"]), left_on="cited_person", right_on="person_title", how="inner")
        cites_by_year = cites_with_birth.filter(pl.col("birth_year").is_not_null()).group_by(["birth_year", "gender"]).len().collect(engine="streaming").to_pandas()
        cites_by_year = cites_by_year[cites_by_year["birth_year"] > 1800] 
        fig_time = px.scatter(cites_by_year, x="birth_year", y="len", color="gender", trendline="lowess", title="Average Citations by Birthdate", color_discrete_map=GENDER_COLORS)
        st.plotly_chart(fig_time, use_container_width=True)

elif page == "Sentiment & Rank Analysis":
    st.header("Sentiment & Rank Analysis")
    df_sentiment = load_sentiment()
    if not df_sentiment.is_empty() and lazy_cites is not None:
        cs_cites_filtered = lazy_cites.filter(pl.col("source_page").is_in(list(cs_topics)))
        cs_cites = cs_cites_filtered.join(df_cs_people.lazy().select("person_title"), left_on="cited_person", right_on="person_title", how="inner")
        cite_counts = cs_cites.group_by(["cited_person", "gender"]).len().sort("len", descending=True).collect(engine="streaming")
        
        top_men = cite_counts.filter(pl.col("gender") == "male").head(100)
        top_women = cite_counts.filter(pl.col("gender") == "female").head(100)
        top_100_combined = pl.concat([top_men, top_women])
        
        # 1. Join sentiment directly with genders to get full coverage
        sent_joined = df_sentiment.join(df_cs_people.select(["person_title", "gender"]), left_on="cited_entity", right_on="person_title", how="inner").to_pandas()
        
        # 2. Force the Top 10 to be the exact Top 10 from the global citation count
        top_10_men_list = top_men.head(10)['cited_person'].to_list()
        top_10_women_list = top_women.head(10)['cited_person'].to_list()
        
        # Filter sentiment to just these top people
        sent_top_men = sent_joined[sent_joined['cited_entity'].isin(top_10_men_list)]
        sent_top_women = sent_joined[sent_joined['cited_entity'].isin(top_10_women_list)]
        
        def calculate_ratios(df, top_list):
            dist = df.groupby(['cited_entity', 'sentiment']).size().unstack(fill_value=0).reset_index()
            for col in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
                if col not in dist.columns:
                    dist[col] = 0
            dist['total'] = dist['POSITIVE'] + dist['NEGATIVE'] + dist['NEUTRAL']
            dist['% POSITIVE'] = (dist['POSITIVE'] / dist['total']) * 100
            dist['% NEGATIVE'] = (dist['NEGATIVE'] / dist['total']) * 100
            dist['% NEUTRAL'] = (dist['NEUTRAL'] / dist['total']) * 100
            # Sort by the original order in top_list
            dist['cited_entity'] = pd.Categorical(dist['cited_entity'], categories=top_list, ordered=True)
            dist = dist.sort_values('cited_entity')
            return pd.melt(dist, id_vars=['cited_entity'], value_vars=['% POSITIVE', '% NEUTRAL', '% NEGATIVE'], var_name='sentiment', value_name='percentage')

        st.subheader("Sentiment Breakdown for Top 10 Individuals")
        
        # Men Stacked Bar
        men_ratios = calculate_ratios(sent_top_men, top_10_men_list)
        men_ratios = men_ratios[men_ratios['sentiment'].isin(['% POSITIVE', '% NEGATIVE', '% NEUTRAL'])]
        fig_men = px.bar(men_ratios, x="cited_entity", y="percentage", color="sentiment", 
                         title="Top 10 Men: Sentiment Breakdown", barmode="stack",
                         color_discrete_map={"% POSITIVE": "#00CC96", "% NEGATIVE": "#EF553B", "% NEUTRAL": "#636EFA"},
                         text=men_ratios['percentage'].apply(lambda x: f"{x:.1f}%" if x > 5 else ""))
        fig_men.update_traces(textposition='inside')
        st.plotly_chart(fig_men, use_container_width=True)
        
        # Women Stacked Bar
        women_ratios = calculate_ratios(sent_top_women, top_10_women_list)
        women_ratios = women_ratios[women_ratios['sentiment'].isin(['% POSITIVE', '% NEGATIVE', '% NEUTRAL'])]
        fig_women = px.bar(women_ratios, x="cited_entity", y="percentage", color="sentiment", 
                           title="Top 10 Women: Sentiment Breakdown", barmode="stack",
                           color_discrete_map={"% POSITIVE": "#00CC96", "% NEGATIVE": "#EF553B", "% NEUTRAL": "#636EFA"},
                           text=women_ratios['percentage'].apply(lambda x: f"{x:.1f}%" if x > 5 else ""))
        fig_women.update_traces(textposition='inside')
        st.plotly_chart(fig_women, use_container_width=True)
        
        # Original Gender Ratio Chart
        sent_dist = sent_joined.groupby(["gender", "sentiment"]).size().reset_index(name='len')
        # Calculate the percentage within each gender
        sent_dist['percentage'] = sent_dist.groupby('gender')['len'].transform(lambda x: (x / x.sum()) * 100)
        
        fig_sent = px.bar(sent_dist, x="gender", y="percentage", color="sentiment", 
                          title="Citation Sentiment Ratio (%) by Gender", barmode="group",
                          text=sent_dist['percentage'].apply(lambda x: f'{x:.1f}%'),
                          color_discrete_map={"POSITIVE": "#00CC96", "NEGATIVE": "#EF553B", "NEUTRAL": "#636EFA"})
        fig_sent.update_traces(textposition='outside')
        fig_sent.update_layout(yaxis_title="Percentage (%)", yaxis_range=[0, 110])
        st.plotly_chart(fig_sent, use_container_width=True)
        
        # 3. Event Quality (Positive vs Negative Events)
        if "event_quality" in df_sentiment.columns:
            st.subheader("Information Quality Breakdown for Top 10 Individuals")
            
            def calculate_event_ratios(df, top_list):
                dist = df.groupby(['cited_entity', 'event_quality']).size().unstack(fill_value=0).reset_index()
                for col in ['POSITIVE', 'NEGATIVE']:
                    if col not in dist.columns:
                        dist[col] = 0
                dist['total'] = dist['POSITIVE'] + dist['NEGATIVE']
                dist['% POSITIVE'] = dist.apply(lambda row: (row['POSITIVE'] / row['total']) * 100 if row['total'] > 0 else 0, axis=1)
                dist['% NEGATIVE'] = dist.apply(lambda row: (row['NEGATIVE'] / row['total']) * 100 if row['total'] > 0 else 0, axis=1)
                dist['cited_entity'] = pd.Categorical(dist['cited_entity'], categories=top_list, ordered=True)
                dist = dist.sort_values('cited_entity')
                return pd.melt(dist, id_vars=['cited_entity'], value_vars=['% POSITIVE', '% NEGATIVE'], var_name='event_quality', value_name='percentage')

            men_event_ratios = calculate_event_ratios(sent_top_men, top_10_men_list)
            fig_men_event = px.bar(men_event_ratios, x="cited_entity", y="percentage", color="event_quality", 
                             title="Top 10 Men: Event Quality Breakdown", barmode="stack",
                             color_discrete_map={"% POSITIVE": "#00CC96", "% NEGATIVE": "#EF553B"},
                             text=men_event_ratios['percentage'].apply(lambda x: f"{x:.1f}%" if x > 5 else ""))
            fig_men_event.update_traces(textposition='inside')
            st.plotly_chart(fig_men_event, use_container_width=True)
            
            women_event_ratios = calculate_event_ratios(sent_top_women, top_10_women_list)
            fig_women_event = px.bar(women_event_ratios, x="cited_entity", y="percentage", color="event_quality", 
                               title="Top 10 Women: Event Quality Breakdown", barmode="stack",
                               color_discrete_map={"% POSITIVE": "#00CC96", "% NEGATIVE": "#EF553B"},
                               text=women_event_ratios['percentage'].apply(lambda x: f"{x:.1f}%" if x > 5 else ""))
            fig_women_event.update_traces(textposition='inside')
            st.plotly_chart(fig_women_event, use_container_width=True)

            st.subheader("Global Information Quality Ratio (%) by Gender")
            event_dist = sent_joined.groupby(["gender", "event_quality"]).size().reset_index(name='len')
            event_dist['percentage'] = event_dist.groupby('gender')['len'].transform(lambda x: (x / x.sum()) * 100)
            
            fig_event = px.bar(event_dist, x="gender", y="percentage", color="event_quality", 
                              title="Information Quality Ratio (%) by Gender", barmode="group",
                              text=event_dist['percentage'].apply(lambda x: f'{x:.1f}%'),
                              color_discrete_map={"POSITIVE": "#00CC96", "NEGATIVE": "#EF553B"})
            fig_event.update_traces(textposition='outside')
            fig_event.update_layout(yaxis_title="Percentage (%)", yaxis_range=[0, 110])
            st.plotly_chart(fig_event, use_container_width=True)
            
            st.info("The Sentiment chart measures Author Tone (praising vs critical language), while the Information Quality chart measures Life Events (awards/successes vs failures/deaths).")

elif page == "Galaxy Network Plots":
    st.header("Galaxy Network Plots")
    st.write("Generating networks... limiting to top 300 connections for rendering performance.")
    if lazy_cites is not None:
        # P2P Lazy Eval
        p2p_edges = lazy_cites.filter(pl.col("source_page").is_in(list(df_cs_people["person_title"]))).group_by(["source_page", "cited_person"]).len().sort("len", descending=True).head(300).collect(engine="streaming").to_pandas()
        
        G_p2p = nx.from_pandas_edgelist(p2p_edges, 'source_page', 'cited_person', edge_attr='len')
        pos_p2p = nx.spring_layout(G_p2p, k=0.15, iterations=20)
        
        edge_x, edge_y = [], []
        for edge in G_p2p.edges():
            x0, y0 = pos_p2p[edge[0]]
            x1, y1 = pos_p2p[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_text, node_adj = [], [], [], []
        for node, adjacencies in enumerate(G_p2p.adjacency()):
            x, y = pos_p2p[adjacencies[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(adjacencies[0])
            node_adj.append(len(adjacencies[1]))
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', textposition="top center", hoverinfo='text', text=node_text,
            marker=dict(showscale=True, colorscale='Plasma', size=10, color=node_adj, line_width=2))
            
        fig_net1 = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title='People  People Citations Galaxy', showlegend=False, hovermode='closest', plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig_net1, use_container_width=True)
        
        # P2T Lazy Eval
        p2t_edges = lazy_cites.filter(pl.col("source_page").is_in(list(cs_topics))).group_by(["source_page", "cited_person"]).len().sort("len", descending=True).head(300).collect(engine="streaming").to_pandas()
        
        G_p2t = nx.from_pandas_edgelist(p2t_edges, 'source_page', 'cited_person', edge_attr='len')
        pos_p2t = nx.spring_layout(G_p2t, k=0.15, iterations=20)
        
        edge_x, edge_y = [], []
        for edge in G_p2t.edges():
            x0, y0 = pos_p2t[edge[0]]
            x1, y1 = pos_p2t[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
        node_x, node_y, node_text, node_adj = [], [], [], []
        for node, adjacencies in enumerate(G_p2t.adjacency()):
            x, y = pos_p2t[adjacencies[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(adjacencies[0])
            node_adj.append(len(adjacencies[1]))
            
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', textposition="top center", hoverinfo='text', text=node_text,
            marker=dict(showscale=True, colorscale='Viridis', size=10, color=node_adj, line_width=2))
            
        fig_net2 = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            title='CS Topics  People Citations Galaxy', showlegend=False, hovermode='closest', plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        st.plotly_chart(fig_net2, use_container_width=True)

# =============================================================================
# PAGE: Language Bias (Phase 7)
# =============================================================================
elif page == "Language Bias (Phase 7)":
    st.header("Advanced Language Bias Analysis (Phase 7)")
    st.markdown("""
    This analysis applies **spaCy POS tagging**, **Named Entity Recognition**, and **Word2Vec embeddings**
    exclusively to Computer Science biographies (7,537 people - 6,058 men, 1,479 women).
    Run `phase7_experimental.py` to generate or refresh these results.
    """)

    ADJ_PATH  = os.path.join("output", "phase7_adjectives.parquet")
    ORG_PATH  = os.path.join("output", "phase7_organizations.parquet")
    BIAS_PATH = os.path.join("output", "phase7_word2vec_bias.parquet")

    any_missing = not (os.path.exists(ADJ_PATH) and os.path.exists(ORG_PATH) and os.path.exists(BIAS_PATH))
    if any_missing:
        st.warning("Phase 7 output files not found. Please run `phase7_experimental.py` first.")
        st.stop()

    df_adj  = pl.read_parquet(ADJ_PATH).to_pandas()
    df_org  = pl.read_parquet(ORG_PATH).to_pandas()
    df_bias = pl.read_parquet(BIAS_PATH).to_pandas()


    # ---- 1. Adjective Proximity ----
    st.subheader("1. Adjective Proximity (within 5 words of a gendered pronoun)")
    st.markdown("""
    Adjectives extracted within a 5-word window of *he/his/him* vs *she/her* pronouns.
    The Y-axis shows the **percentage of people** in each group whose biography contains that adjective
    near a gendered pronoun - making the comparison fair despite the 4:1 gender imbalance in the dataset.
    """)

    top_n_adj = st.slider("Number of top adjectives to display", 5, 30, 15, key="adj_slider")

    col_m, col_f = st.columns(2)
    with col_m:
        df_m = df_adj.nlargest(top_n_adj, "male_pct")[["word", "male_pct"]].sort_values("male_pct")
        fig_m = px.bar(
            df_m, x="male_pct", y="word", orientation="h",
            title="Top Adjectives near Male Pronouns",
            color_discrete_sequence=[GENDER_COLORS["Male"]],
            labels={"male_pct": "% of men with this adjective nearby", "word": "Adjective"}
        )
        fig_m.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_m, use_container_width=True)

    with col_f:
        df_f = df_adj.nlargest(top_n_adj, "female_pct")[["word", "female_pct"]].sort_values("female_pct")
        fig_f = px.bar(
            df_f, x="female_pct", y="word", orientation="h",
            title="Top Adjectives near Female Pronouns",
            color_discrete_sequence=[GENDER_COLORS["Female"]],
            labels={"female_pct": "% of women with this adjective nearby", "word": "Adjective"}
        )
        fig_f.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")

    # ---- 2. NER Organizations ----
    st.subheader("2. Named Entity Recognition - Organizations")

    col_exp, col_ctrl = st.columns([3, 1])
    with col_exp:
        st.markdown("""
        Two views of the same data:
       - **Percentage view** (left/right): share of biographies per gender mentioning this organization, normalized to be comparable.
       - **Absolute comparison** (below): raw mention counts side-by-side so you can see the true volume difference.
        """)
    with col_ctrl:
        top_n_org = st.slider("Top N organizations", 5, 20, 10, key="org_slider")

    # --- Percentage side-by-side ---
    col_om, col_of = st.columns(2)
    with col_om:
        df_om = df_org.nlargest(top_n_org, "male_pct")[["org", "male_pct"]].sort_values("male_pct")
        fig_om = px.bar(
            df_om, x="male_pct", y="org", orientation="h",
            title="Top Organizations in Male Biographies (%)",
            color_discrete_sequence=[GENDER_COLORS["Male"]],
            labels={"male_pct": "% of men mentioning this org", "org": "Organization"}
        )
        fig_om.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_om, use_container_width=True)

    with col_of:
        df_of = df_org.nlargest(top_n_org, "female_pct")[["org", "female_pct"]].sort_values("female_pct")
        fig_of = px.bar(
            df_of, x="female_pct", y="org", orientation="h",
            title="Top Organizations in Female Biographies (%)",
            color_discrete_sequence=[GENDER_COLORS["Female"]],
            labels={"female_pct": "% of women mentioning this org", "org": "Organization"}
        )
        fig_of.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_of, use_container_width=True)

    # --- Grouped absolute comparison ---
    st.markdown("**Absolute mention counts - all organizations, both genders side by side:**")
    df_org_sorted = df_org.sort_values("male_count", ascending=False).head(top_n_org)
    df_org_melt = df_org_sorted[["org", "male_count", "female_count"]].melt(
        id_vars="org", var_name="Gender", value_name="Count"
    )
    df_org_melt["Gender"] = df_org_melt["Gender"].map({"male_count": "Male", "female_count": "Female"})
    fig_grouped = px.bar(
        df_org_melt, x="org", y="Count", color="Gender", barmode="group",
        color_discrete_map=GENDER_COLORS,
        labels={"org": "Organization", "Count": "Raw Mention Count"},
        title="Absolute Organization Mentions - Male vs Female (raw counts)"
    )
    fig_grouped.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-30
    )
    st.plotly_chart(fig_grouped, use_container_width=True)

    st.markdown("---")

    # ---- 3. Word2Vec Bias ----
    st.subheader("3. Word2Vec Language Bias Scores")

    with st.expander("How is this score calculated?"):
        st.markdown("""
        **Word2Vec** is trained on all 192,852 sentences from the CS biographies. It learns a
        300-dimensional mathematical vector for every word in the vocabulary by predicting which words
        tend to appear near each other in a sliding 5-word window.

        The **Bias Score** (the horizontal bar length) is computed as:

        ```
        bias_score = avg(cosine_similarity(word, ["he", "him"]))
                  - avg(cosine_similarity(word, ["she", "her"]))
        ```

        **Cosine similarity** measures how geometrically close two word vectors are in that 300D space
        (1.0 = identical direction, -1.0 = opposite direction, 0 = no relationship).

        So a **positive bias score** means the word's vector points closer to male pronouns than female
        pronouns in the learned Wikipedia language model - indicating the word is more linguistically
        associated with men in CS biographies. A **negative score** means the opposite.

        A threshold of ±0.005 is used to classify a word as skewed; anything between is labelled Neutral.
        """)

    st.markdown("""
    A **positive bar (blue)** = this concept is more linguistically associated with male pronouns in Wikipedia CS text.
    A **negative bar (pink)** = more associated with female pronouns.
    """)

    df_bias_sorted = df_bias.sort_values("diff", ascending=False)
    color_map = {"Male skewed": "#4fc3f7", "Female skewed": "#f48fb1", "Neutral": "#aaaaaa"}
    colors = df_bias_sorted["bias"].map(color_map)

    fig_bias = go.Figure(go.Bar(
        x=df_bias_sorted["diff"],
        y=df_bias_sorted["word"],
        orientation="h",
        marker_color=colors,
        text=df_bias_sorted["bias"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Male sim: %{customdata[0]:.4f}<br>Female sim: %{customdata[1]:.4f}<br>Bias score: %{x:.4f}<extra></extra>",
        customdata=df_bias_sorted[["male_sim", "female_sim"]].values
    ))
    fig_bias.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
    fig_bias.update_layout(
        title="Word2Vec Bias Score  (Male cosine sim) - (Female cosine sim)",
        xaxis_title="Bias Score (positive = Male skewed | negative = Female skewed)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        height=520
    )
    st.plotly_chart(fig_bias, use_container_width=True)

    st.markdown("**Detailed similarity scores per word:**")
    df_display = df_bias_sorted[["word", "male_sim", "female_sim", "diff", "bias"]].copy()
    df_display.columns = ["Word", "Male Similarity", "Female Similarity", "Bias Score", "Direction"]
    st.dataframe(
        df_display.style
            .background_gradient(subset=["Bias Score"], cmap="RdBu", vmin=-0.2, vmax=0.2)
            .format({"Male Similarity": "{:.4f}", "Female Similarity": "{:.4f}", "Bias Score": "{:.4f}"}),
        use_container_width=True
    )
