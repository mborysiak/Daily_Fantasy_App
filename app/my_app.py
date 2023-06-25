import streamlit as st

def main():
    st.title("My First Streamlit App")
    st.write("Welcome to my app!")

    button_clicked = st.button("Click me!")

    if button_clicked:
        st.write("Button clicked!")

if __name__ == '__main__':
    main()