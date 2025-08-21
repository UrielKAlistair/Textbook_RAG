from preprocessing.parse_input import parse_pdf
from preprocessing.generate_embeddings import generate_embeddings

file_name = "Introduction_to_probability-223-266.pdf"
parse_pdf(file_name)
generate_embeddings(file_name)

# Retrieve relevant vectors, pick out parent sections, augment prompt and present as output.
# This has to be made a fastapi endpoint to present the results, we can call it from a frontend at a later date.
# We should probably put the above preprocessing thing under an if or comment it out.
# If we establish multiple datasets, we can provide frontend choice for it and fetch from corresponding vectorDB.
