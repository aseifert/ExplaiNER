import matplotlib as matplotlib
import matplotlib.cm as cm
import pandas as pd
import streamlit as st
import tokenizers
import torch
import torch.nn.functional as F
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

tokenizer_hash_funcs = {
    tokenizers.Tokenizer: lambda _: None,
    tokenizers.AddedToken: lambda _: None,
}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu" if torch.has_mps else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classmap = {
    "O": "O",
    "PER": "🙎",
    "person": "🙎",
    "LOC": "🌎",
    "location": "🌎",
    "ORG": "🏤",
    "corporation": "🏤",
    "product": "📱",
    "creative": "🎷",
    "MISC": "🎷",
}


def aggrid_interactive_table(df: pd.DataFrame) -> dict:
    """Creates an st-aggrid interactive table based on a dataframe.
    Args:
        df (pd.DataFrame]): Source dataframe
    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()
    # options.configure_default_column(cellRenderer=JsCode('''function(params) {return '<a href="#samples-loss">'+params.value+'</a>'}'''))

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
    )

    return selection


def explode_df(df: pd.DataFrame) -> pd.DataFrame:
    df_tokens = df.apply(pd.Series.explode)
    if "losses" in df.columns:
        df_tokens["losses"] = df_tokens["losses"].astype(float)
    return df_tokens  # type: ignore


def align_sample(row: pd.Series):
    """Use word_ids to align all lists in a sample."""

    columns = row.axes[0].to_list()
    indices = [i for i, id in enumerate(row.word_ids) if id >= 0 and id != row.word_ids[i - 1]]

    out = {}

    tokens = []
    for i, tok in enumerate(row.tokens):
        if row.word_ids[i] == -1:
            continue

        if row.word_ids[i] != row.word_ids[i - 1]:
            tokens.append(tok.lstrip("▁").lstrip("##").rstrip("@@"))
        else:
            tokens[-1] += tok.lstrip("▁").lstrip("##").rstrip("@@")
    out["tokens"] = tokens

    if "labels" in columns:
        out["labels"] = [row.labels[i] for i in indices]

    if "preds" in columns:
        out["preds"] = [row.preds[i] for i in indices]

    if "losses" in columns:
        out["losses"] = [row.losses[i] for i in indices]

    if "probs" in columns:
        out["probs"] = [row.probs[i] for i in indices]

    if "hidden_states" in columns:
        out["hidden_states"] = [row.hidden_states[i] for i in indices]

    if "ids" in columns:
        out["ids"] = row.ids

    assert len(tokens) == len(out["preds"]), (tokens, row.tokens)

    return out


@st.cache(
    allow_output_mutation=True,
    hash_funcs=tokenizer_hash_funcs,
)
def tag_text(text: str, tokenizer, model, device: torch.device) -> pd.DataFrame:
    """Create an (exploded) DataFrame with the predicted labels and probabilities."""

    tokens = tokenizer(text).tokens()
    tokenized = tokenizer(text, return_tensors="pt")
    word_ids = [w if w is not None else -1 for w in tokenized.word_ids()]
    input_ids = tokenized.input_ids.to(device)
    outputs = model(input_ids, output_hidden_states=True)
    preds = torch.argmax(outputs.logits, dim=2)
    preds = [model.config.id2label[p] for p in preds[0].cpu().numpy()]
    hidden_states = outputs.hidden_states[-1][0].detach().cpu().numpy()
    # hidden_states = np.mean([hidden_states, outputs.hidden_states[0][0].detach().cpu().numpy()], axis=0)

    probs = 1 // (
        torch.min(F.softmax(outputs.logits, dim=-1), dim=-1).values[0].detach().cpu().numpy()
    )

    df = pd.DataFrame(
        [[tokens, word_ids, preds, probs, hidden_states]],
        columns="tokens word_ids preds probs hidden_states".split(),
    )
    merged_df = pd.DataFrame(df.apply(align_sample, axis=1).tolist())
    return explode_df(merged_df).reset_index().drop(columns=["index"])


def get_bg_color(label):
    return st.session_state[f"color_{label}"]


def get_fg_color(hex_color: str) -> str:
    """Adapted from https://gomakethings.com/dynamically-changing-the-text-color-based-on-background-color-contrast-with-vanilla-js/"""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000
    return "black" if (yiq >= 128) else "white"


def colorize_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Colorize the errors in the dataframe."""

    def colorize_row(row):
        return [
            "background-color: "
            + ("white" if (row["labels"] == "IGN" or (row["preds"] == row["labels"])) else "pink")
            + ";"
        ] * len(row)

    def colorize_col(col):
        if col.name == "labels" or col.name == "preds":
            bgs = []
            fgs = []
            for v in col.values:
                bgs.append(get_bg_color(v.split("-")[1]) if "-" in v else "#ffffff")
                fgs.append(get_fg_color(bgs[-1]))
            return [f"background-color: {bg}; color: {fg};" for bg, fg in zip(bgs, fgs)]
        return [""] * len(col)

    df = df.reset_index().drop(columns=["index"]).T
    return df  # .style.apply(colorize_col, axis=0)


def htmlify_labeled_example(example: pd.DataFrame) -> str:
    html = []

    for _, row in example.iterrows():
        pred = row.preds.split("-")[1] if "-" in row.preds else "O"
        label = row.labels
        label_class = row.labels.split("-")[1] if "-" in row.labels else "O"

        color = get_bg_color(row.preds.split("-")[1]) if "-" in row.preds else "#000000"
        true_color = get_bg_color(row.labels.split("-")[1]) if "-" in row.labels else "#000000"

        font_color = get_fg_color(color) if color else "white"
        true_font_color = get_fg_color(true_color) if true_color else "white"

        is_correct = row.preds == row.labels
        loss_html = (
            ""
            if float(row.losses) < 0.01
            else f"<span style='background-color: yellow; color: font_color; padding: 0 5px;'>{row.losses:.3f}</span>"
        )
        loss_html = ""

        if row.labels == row.preds == "O":
            html.append(f"<span>{row.tokens}</span>")
        elif row.labels == "IGN":
            assert False
        else:
            opacity = "1" if not is_correct else "0.5"
            correct = (
                ""
                if is_correct
                else f"<span title='{label}' style='background-color: {true_color}; opacity: 1; color: {true_font_color}; padding: 0 5px; border: 1px solid black; min-width: 30px'>{classmap[label_class]}</span>"
            )
            pred_icon = classmap[pred] if pred != "O" and row.preds[:2] != "I-" else ""
            html.append(
                f"<span style='border: 1px solid black; color: {color}; padding: 0 5px;' title={row.preds}>{pred_icon + ' '}{row.tokens}</span>{correct}{loss_html}"
            )

    return " ".join(html)


def htmlify_example(example: pd.DataFrame) -> str:
    corr_html = " ".join(
        [
            f", {row.tokens}" if row.labels == "B-COMMA" else row.tokens
            for _, row in example.iterrows()
        ]
    ).strip()
    return f"<em>{corr_html}</em>"


def color_map_color(value: float, cmap_name="Set1", vmin=0, vmax=1) -> str:
    """Turn a value into a color using a color map."""
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgba = cmap(norm(abs(value)))
    color = matplotlib.colors.rgb2hex(rgba[:3])
    return color
