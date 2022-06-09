# ExplaiNER

Error Analysis is an important but often overlooked part of the data science project lifecycle, for which there is still very little tooling available. Due to the lack of tooling, practitioners often write throwaway code or, worse, skip understanding their models' errors altogether. This project tries to provide an extensive toolkit to probe any NER model/dataset combination, find labeling errors and understand the models' and datasets' limitations, leading the user on her way to further improvements.


## Sections

### Probing

A very direct and interactive way to test your model is by providing it with a list of text inputs and then inspecting the model outputs. The application features a multiline text field so the user can input multiple texts separated by newlines. For each text, the app will show a data frame containing the tokenized string, token predictions, probabilities and a visual indicator for low probability predictions -- these are the ones you should inspect first for prediction errors.

### Embeddings

For every token in the dataset, we take its hidden state and using TruncatedSVD we project it onto a two-dimensional plane. Data points are colored by label, with mislabeled examples signified by a small black border.

### Metrics

The metrics page contains precision, recall and f-score metrics as well as a confusion matrix over all the classes. By default, the confusion matrix is normalized. There's an option to zero out the diagonal, leaving only prediction errors (here it makes sense to turn off normalization, so you get raw error counts).

### Misclassified

asdf

### Loss by Token/Label

Shows count, mean and median loss per token and label.

### Samples by Loss

Shows every example sorted by loss (descending) for close inspection.

### Random Samples

Shows random samples. Simple idea, but often it turns up some interesting things.

### Inspect

Inspect your whole dataset, either unfiltered or by id.

### Raw data

See the data as seen by your model.

### Debug

Some debug info.
