import streamlit as st
from hmm import tagger
import pandas as pd

# Streamlit App
def main():
    st.title("HMM-Viterbi POS Tagging")

    st.write("Enter a sentence to see its POS tags:")
    
    sentence = st.text_input("Sentence", "The quick brown fox jumps over the lazy dog")
    
    words = sentence.split(' ')
    if '' in words:
        words.remove('')


    # Call the Viterbi algorithm function
    tagged_sentence = tagger.tag(words)

    print('words:', len(words), words)
    print('tagged_sentence:', len(tagged_sentence), tagged_sentence)
    print()
    
    # Display the tagged sentence
    st.write("Tagged Sentence:")

    df = pd.DataFrame(list(zip(words, tagged_sentence)), columns=['Word', 'POS Tag'])

    st.dataframe(df, use_container_width=True)

    # st.write(tagged_sentence)

if __name__ == "__main__":
    main()
