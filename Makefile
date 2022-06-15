doc:
	pdoc --docformat google src -o doc

vis2: doc
	pandoc html/index.md -s -o html/index.html
	rm -rf src/__pycache__ && rm -rf src/subpages/__pycache__
	zip -r vis2.zip doc html src Makefile presentation.pdf requirements.txt

run:
	python -m streamlit run src/app.py
