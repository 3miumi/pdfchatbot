import streamlit as st
from st_pages import Page, add_page_title, get_nav_from_toml

st.set_page_config()

nav = get_nav_from_toml(
".streamlit/pages.toml"
)
pg = st.navigation(nav)

add_page_title(pg)

pg.run()