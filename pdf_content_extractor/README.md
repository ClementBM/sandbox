- [Python PDF Content Extractor](#python-pdf-content-extractor)
  - [Librairies](#librairies)
    - [python-docx](#python-docx)
    - [pypdf](#pypdf)
    - [pdfminer-six](#pdfminer-six)
    - [pdf2docx](#pdf2docx)
    - [Apache Tika](#apache-tika)
    - [pdfplumber](#pdfplumber)
    - [`soffice` linux package](#soffice-linux-package)
  - [Evaluation](#evaluation)
    - [pdf-text-extraction-benchmark](#pdf-text-extraction-benchmark)
    - [py-pdf benchmarks](#py-pdf-benchmarks)
  - [EPUB Format](#epub-format)
    - [ebooklib](#ebooklib)
  - [Evaluation Data](#evaluation-data)


# Python PDF Content Extractor
Portable Document Format (PDF) files are commonly used for sharing documents electronically. Individuals and businesses use PDF files to share information alike. Often we need to extract some information from the PDF files for further processing. However, extracting text from a PDF file can be challenging, especially if the document contains complex formatting and layout. Fortunately, there are several ways to do this.

Here, we will provide the most commonly used method to extract text from PDFs using Python. Python comprises several libraries that enable efficient PDF text extraction.

The article explores some popular Python libraries for extracting text from PDF files and the step-by-step text extraction process from PDFs.

* from https://nanonets.com/blog/extract-text-from-pdf-file-using-python/
## Librairies

### [python-docx](https://github.com/python-openxml/python-docx)
`python-docx` is a Python library for creating and updating Microsoft Word (.docx) files.

### [pypdf](https://github.com/py-pdf/pypdf)
pypdf is a free and open-source pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files. It can also add custom data, viewing options, and passwords to PDF files. pypdf can retrieve text and metadata from PDFs as well.

### [pdfminer-six](https://github.com/pdfminer/pdfminer.six/blob/master/README.md)
Pdfminer.six is a community maintained fork of the original PDFMiner. It is a tool for extracting information from PDF documents. It focuses on getting and analyzing text data. Pdfminer.six extracts the text from a page directly from the sourcecode of the PDF. It can also be used to get the exact location, font or color of the text.

### [pdf2docx](https://github.com/dothinking/pdf2docx)
* Extract data from PDF with PyMuPDF, e.g. text, images and drawings
* Parse layout with rule, e.g. sections, paragraphs, images and tables
* Generate docx with python-docx

### [Apache Tika](https://github.com/chrismattmann/tika-python)
A Python port of the Apache Tika library that makes Tika available using the Tika REST Server.

This makes Apache Tika available as a Python library, installable via Setuptools, Pip and Easy Install.

To use this library, you need to have Java 7+ installed on your system as tika-python starts up the Tika REST server in the background.

### [pdfplumber](https://github.com/jsvine/pdfplumber)
Plumb a PDF for detailed information about each text character, rectangle, and line. Plus: Table extraction and visual debugging.

Works best on machine-generated, rather than scanned, PDFs. Built on pdfminer.six.

### `soffice` linux package

## Evaluation

### [pdf-text-extraction-benchmark](https://github.com/ckorzen/pdf-text-extraction-benchmark)
A project about benchmarking and evaluating existing PDF extraction tools on their semantic abilities to extract the body texts from PDF documents, especially from scientific articles.

This project is about benchmarking and evaluating existing PDF extraction tools on their semantic abilities to extract the body texts from PDF documents, especially from scientific articles.

It provides:
1. a benchmark generator,
2. a ready-to-use benchmark and
3. an extensive evaluation, with meaningful evaluation criteria.

### [py-pdf benchmarks](https://github.com/py-pdf/benchmarks)
This benchmark is about reading pure PDF files - notscanned documents and not documents that applied OCR.

## EPUB Format
An .epub file is a zip-encoded file containing a META-INF directory, which contains a file named container.xml, which points to another file usually named Content.opf, which indexes all the other files which make up the e-book (summary based on http://www.jedisaber.com/eBooks/tutorial.asp ; full spec at http://www.idpf.org/2007/opf/opf2.0/download/ )

DRM-free epubs are actually simply renamed zip files. You can change the .epub extension to .zip and unzip like any normal zip file.

Looking at the inside of an epub file, you can see that it mostly contains html files. The text of an epub is stored inside html tags, similar to a website. So, in addition to ebooklib, you can use BeautifulSoup to parse some of the data.

The epub is broken up into different parts, often around chapters in the book. Tables of contents, covers, copyright pages, etc. also often get their own sections in an epub. After loading the epub file into ebooklib, you’ll want to get every item in the epub.

The following Python code will extract the basic meta-information from an .epub file and return it as a dict.

```python
import zipfile
from lxml import etree

def epub_info(fname):
    def xpath(element, path):
        return element.xpath(
            path,
            namespaces={
                "n": "urn:oasis:names:tc:opendocument:xmlns:container",
                "pkg": "http://www.idpf.org/2007/opf",
                "dc": "http://purl.org/dc/elements/1.1/",
            },
        )[0]

    # prepare to read from the .epub file
    zip_content = zipfile.ZipFile(fname)
      
    # find the contents metafile
    cfname = xpath(
        etree.fromstring(zip_content.read("META-INF/container.xml")),
        "n:rootfiles/n:rootfile/@full-path",
    ) 
    
    # grab the metadata block from the contents metafile
    metadata = xpath(
        etree.fromstring(zip_content.read(cfname)), "/pkg:package/pkg:metadata"
    )
    
    # repackage the data
    return {
        s: xpath(metadata, f"dc:{s}/text()")
        for s in ("title", "language", "creator", "date", "identifier")
    }    
```

### [ebooklib](https://pypi.org/project/EbookLib/)
```python
import ebooklib
from ebooklib import epubbook = epub.read_epub(file_name)
items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
```

https://andrew-muller.medium.com/getting-text-from-epub-files-in-python-fbfe5df5c2da

## Evaluation Data
https://books.openedition.org/dice/11013#text 