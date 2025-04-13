create-conda:
	conda env create -f env.yaml

run:
	streamlit run app.py