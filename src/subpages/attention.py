import ecco
import streamlit as st
from streamlit.components.v1 import html

from src.subpages.page import Context, Page  # type: ignore

SETUP_HTML = """
<script src="https://requirejs.org/docs/release/2.3.6/minified/require.js"></script>
<script>
    var ecco_url = 'https://storage.googleapis.com/ml-intro/ecco/'
    //var ecco_url = 'http://localhost:8000/'

    if (window.ecco === undefined) window.ecco = {}

    // Setup the paths of the script we'll be using
    requirejs.config({
        urlArgs: "bust=" + (new Date()).getTime(),
        nodeRequire: require,
        paths: {
            d3: "https://d3js.org/d3.v6.min", // This is only for use in setup.html and basic.html
            "d3-array": "https://d3js.org/d3-array.v2.min",
            jquery: "https://code.jquery.com/jquery-3.5.1.min",
            ecco: ecco_url + 'js/0.0.6/ecco-bundle.min',
            xregexp: 'https://cdnjs.cloudflare.com/ajax/libs/xregexp/3.2.0/xregexp-all.min'
        }
    });

    // Add the css file
    //requirejs(['d3'],
    //    function (d3) {
    //        d3.select('#css').attr('href', ecco_url + 'html/styles.css')
    //    })

    console.log('Ecco initialize!!')

    // returns a 'basic' object. basic.init() selects the html div we'll be
    // rendering the html into, adds styles.css to the document.
    define('basic', ['d3'],
        function (d3) {
            return {
                init: function (viz_id = null) {
                    if (viz_id == null) {
                        viz_id = "viz_" + Math.round(Math.random() * 10000000)
                    }
                    // Select the div rendered below, change its id
                    const div = d3.select('#basic').attr('id', viz_id),
                        div_parent = d3.select('#' + viz_id).node().parentNode

                    // Link to CSS file
                    d3.select(div_parent).insert('link')
                        .attr('rel', 'stylesheet')
                        .attr('type', 'text/css')
                        .attr('href', ecco_url + 'html/0.0.2/styles.css')

                    return viz_id
                }
            }
        }, function (err) {
            console.log(err);
        }
    )
</script>

<head>
    <link id='css' rel="stylesheet" type="text/css">
</head>
<div id="basic"></div>
"""

JS_TEMPLATE = """requirejs(['basic', 'ecco'], function(basic, ecco){{
    const viz_id = basic.init()

    ecco.interactiveTokensAndFactorSparklines(viz_id, {}, {{
    'hltrCFG': {{'tokenization_config': {{'token_prefix': '', 'partial_token_prefix': '##'}}
        }}
    }})
}}, function (err) {{
    console.log(err);
}})"""


@st.cache(allow_output_mutation=True)
def _load_ecco_model():
    model_config = {
        "embedding": "embeddings.word_embeddings",
        "type": "mlm",
        "activations": [r"ffn\.lin1"],
        "token_prefix": "",
        "partial_token_prefix": "##",
    }
    return ecco.from_pretrained(
        "elastic/distilbert-base-uncased-finetuned-conll03-english",
        model_config=model_config,
        activations=True,
    )


class AttentionPage(Page):
    name = "Activations"
    icon = "activity"

    def get_widget_defaults(self):
        return {
            "act_n_components": 8,
            "act_default_text": """Now I ask you: what can be expected of man since he is a being endowed with strange qualities? Shower upon him every earthly blessing, drown him in a sea of happiness, so that nothing but bubbles of bliss can be seen on the surface; give him economic prosperity, such that he should have nothing else to do but sleep, eat cakes and busy himself with the continuation of his species, and even then out of sheer ingratitude, sheer spite, man would play you some nasty trick. He would even risk his cakes and would deliberately desire the most fatal rubbish, the most uneconomical absurdity, simply to introduce into all this positive good sense his fatal fantastic element. It is just his fantastic dreams, his vulgar folly that he will desire to retain, simply in order to prove to himself--as though that were so necessary-- that men still are men and not the keys of a piano, which the laws of nature threaten to control so completely that soon one will be able to desire nothing but by the calendar. And that is not all: even if man really were nothing but a piano-key, even if this were proved to him by natural science and mathematics, even then he would not become reasonable, but would purposely do something perverse out of simple ingratitude, simply to gain his point. And if he does not find means he will contrive destruction and chaos, will contrive sufferings of all sorts, only to gain his point! He will launch a curse upon the world, and as only man can curse (it is his privilege, the primary distinction between him and other animals), may be by his curse alone he will attain his object--that is, convince himself that he is a man and not a piano-key!""",
            "act_from_layer": 0,
            "act_to_layer": 6,
        }

    def render(self, context: Context):
        st.title(self.name)

        with st.expander("ℹ️", expanded=True):
            st.write(
                "A group of neurons tend to fire in response to commas and other punctuation. Other groups of neurons tend to fire in response to pronouns. Use this visualization to factorize neuron activity in individual FFNN layers or in the entire model."
            )

        lm = _load_ecco_model()

        col1, _, col2 = st.columns([1.5, 0.5, 4])
        with col1:
            st.subheader("Settings")
            n_components = st.slider(
                "#components",
                key="act_n_components",
                min_value=2,
                max_value=10,
                step=1,
            )
            from_layer = (
                st.slider(
                    "from layer",
                    key="act_from_layer",
                    value=0,
                    min_value=0,
                    max_value=len(lm.model.transformer.layer) - 1,
                    step=1,
                )
                or None
            )
            to_layer = (
                st.slider(
                    "to layer",
                    key="act_to_layer",
                    value=0,
                    min_value=0,
                    max_value=len(lm.model.transformer.layer),
                    step=1,
                )
                or None
            )
        with col2:
            st.subheader("–")
            text = st.text_area("Text", key="act_default_text")

        inputs = lm.tokenizer([text], return_tensors="pt")
        output = lm(inputs)
        nmf = output.run_nmf(n_components=n_components, from_layer=from_layer, to_layer=to_layer)
        data = nmf.explore(returnData=True)
        JS_TEMPLATE = f"""<script>requirejs(['basic', 'ecco'], function(basic, ecco){{
            const viz_id = basic.init()
            ecco.interactiveTokensAndFactorSparklines(viz_id, {data}, {{ 'hltrCFG': {{'tokenization_config': {{'token_prefix': '', 'partial_token_prefix': '##'}} }} }})
        }}, function (err) {{
            console.log(err);
        }})</script>"""
        html(SETUP_HTML + JS_TEMPLATE, height=800, scrolling=True)
