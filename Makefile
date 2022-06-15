doc:
	pdoc --docformat google src -o doc

vis2: doc
	pandoc html/index.md -s -o html/index.html

run:
	python -m streamlit run src/app.py
