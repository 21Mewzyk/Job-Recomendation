import streamlit as st

# Set sidebar config
def sidebar():
    # st.sidebar.title("About us")
    st.sidebar.subheader("LinkedIn Profile Links:")
    
    text_string_variable1 = "Ryan Lester Pallasigue"
    url_string_variable1 = "https://www.linkedin.com/in/ryan-pallasigue-448339300/"
    link1 = f'[{text_string_variable1}]({url_string_variable1})'
    st.sidebar.markdown(link1, unsafe_allow_html=True)

    text_string_variable2 = "Diosdado Saguiped JR."
    url_string_variable2 = "https://www.linkedin.com/in/diosdado-saguiped-21a9542b5/"
    link2 = f'[{text_string_variable2}]({url_string_variable2})'
    st.sidebar.markdown(link2, unsafe_allow_html=True) 
    
    text_string_variable3 = "Jodel Sawit"
    url_string_variable3 = "https://www.linkedin.com/in/jodel-sawit-596521277/"
    link3 = f'[{text_string_variable3}]({url_string_variable3})'
    st.sidebar.markdown(link3, unsafe_allow_html=True) 

