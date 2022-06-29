"""
A very direct and interactive way to test your model is by providing it with a list of text inputs and then inspecting the model outputs. The application features a multiline text field so the user can input multiple texts separated by newlines. For each text, the app will show a data frame containing the tokenized string, token predictions, probabilities and a visual indicator for low probability predictions -- these are the ones you should inspect first for prediction errors.
"""
import streamlit as st

from src.subpages.page import Context, Page
from src.utils import device, tag_text

_DEFAULT_SENTENCES = """
Damit hatte er auf ihr letztes , völlig schiefgelaufenes Geschäftsessen angespielt .
Damit einher geht übrigens auch , dass Marcella , Collocinis Tochter , keine wie auch immer geartete strafrechtliche Verfolgung zu befürchten hat .
Nach dem Bell ’ schen Theorem , einer Physik jenseits der Quanten , ist die Welt , die wir für real halten , nicht objektivierbar .
Dazu muss man wiederum wissen , dass die Aussagekraft von Tests , neben der Sensitivität und Spezifität , ganz entscheidend von der Vortestwahrscheinlichkeit abhängt .
Haben Sie sich schon eingelebt ? « erkundigte er sich .
Das Auto ein Totalschaden , mein Beifahrer ein weinender Jammerlappen .
Seltsam , wunderte sie sich , dass das Stück nach mehr als eineinhalb Jahrhunderten noch so gut in Schuss ist .
Oder auf den Strich gehen , Strümpfe stricken , Geld hamstern .
Und Allah ist Allumfassend Allwissend .
Und Pedro Moacir redete weiter : » Verzicht , Pater Antonio , Verzicht , zu großer Schmerz über Verzicht , Sehnsucht , die sich nicht erfüllt , die sich nicht erfüllen kann , das sind Qualen , die ein Verstummen nach sich ziehen können , oder Härte .
Mama-San ging mittlerweile fast ausnahmslos nur mit Wei an ihrer Seite aus dem Haus , kaum je mit einem der Mädchen und niemals allein.
""".strip()
_DEFAULT_SENTENCES = """
Elon Musk’s Berghain humiliation — I know the feeling
Musk was also seen at a local spot called Sisyphos celebrating entrepreneur Adeo Ressi's birthday, according to The Times.
""".strip()


class ProbingPage(Page):
    name = "Probing"
    icon = "fonts"

    def _get_widget_defaults(self):
        return {"probing_textarea": _DEFAULT_SENTENCES}

    def render(self, context: Context):
        st.title("🔠 Interactive Probing")

        with st.expander("💡", expanded=True):
            st.write(
                "A very direct and interactive way to test your model is by providing it with a list of text inputs and then inspecting the model outputs. The application features a multiline text field so the user can input multiple texts separated by newlines. For each text, the app will show a data frame containing the tokenized string, token predictions, probabilities and a visual indicator for low probability predictions -- these are the ones you should inspect first for prediction errors."
            )

        sentences = st.text_area("Sentences", height=200, key="probing_textarea")
        if not sentences.strip():
            return
        sentences = [sentence.strip() for sentence in sentences.splitlines()]

        for sent in sentences:
            sent = sent.replace(",", "").replace("  ", " ")
            with st.expander(sent):
                tagged = tag_text(sent, context.tokenizer, context.model, device)
                tagged = tagged.astype(str)
                tagged["probs"] = tagged["probs"].apply(lambda x: x[:-2])
                tagged["check"] = tagged["probs"].apply(
                    lambda x: "✅ ✅" if int(x) < 100 else "✅" if int(x) < 1000 else ""
                )
                st.dataframe(tagged.drop("hidden_states", axis=1).T)
