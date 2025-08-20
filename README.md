# Textbook_RAG
Seeks to make a browser plugin that can augment your prompts to chatGPT based on a thorough RAG tailored to your particular needs.

## Logs:
3 Steps: 1. Parse Input 2. Chunk and Generate embeddings 3. Retrieve and Augment prompt

### Manual Scanning
- Tried to parse text from a pdf with existing OCR. 
- Resulted in text within Image entering main body.
- Detected images, found overlaping text, removed such text from main body, cropped out the image bbox with the text, and used it for generating textual description.
- Found that most images are actually vectors, and they aren't detected as images.
- Realised that manually piecing together vector diagrams in code, then adding text, then dealing with edge cases, other image types, pdfs without OCR etc etc was not worth the effort.
- Found that someone had made a package called docling exactly for this purpose.

Takeaway: Building even a simple PDF parser from scratch is a non-trivial task and tools have been built in the public domain for this express purpose.

### Docling
- docling works fine but takes 20s per page when there are images, 10s when the image detection is turned off.
- Tried to get CUDA to activate and hoped for speedup, still slow.
- Tried running on google colab, was pointless.

Takeaway: Small as the parameter size may be, the inference times for these models is actually obscene. Making API calls is the best bet.

- Decided to integrate cloud services (gemini) for image description. Docling internally constructs post requests using a wrapper and no matter how I modified the options, the API requests to gemini would throw a 404 or a 402 Bad Request. I Decided to locally host an endpoint and make it act as adapter to gemini. and just use the python package gemini has for making calls.

- When I saw the inputs to my local endpoint, I realised the reason things were failing was that the fundamental structure of the request is different: The default API integration for docling is with the OpenAI format of requests, and gemini has another format. I decided that since I'd come this far, I may as well finish it using the gemini python package and call it a day.

- I found that even the response from gemini had to be in the OpenAI format for docling to work. I found that gemini had exposed a separate endpoint for easy integration with code that uses OPENAI format. I deleted the fastapi endpoint I made and questioned my life choices. The Image description calls now worked.

Takeaway: Gemini has OpenAI format endpoints. (Plus invest into investigating why code breaks; I wouldn't have found the OpenAI issue if I hadn't intercepted the request myself; It was lack of expertise however, in not searching immediately for an OpenAI format endpoint under gemini.)


### 