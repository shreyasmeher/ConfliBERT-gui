# ConfliBERT Demo Application

A web-based interface for ConfliBERT, a BERT-based model specialized in conflict and political event analysis. This application provides multiple Natural Language Processing capabilities including Named Entity Recognition (NER), Text Classification, Multi-label Classification, and Question Answering.

## Features

- **Named Entity Recognition (NER)**
  - Identifies and classifies named entities in text
  - Entities include: Organizations, Persons, Locations, Quantities, Weapons, Nationalities, Temporal references, and more
  - Color-coded visualization of entities in the web interface

- **Text Classification**
  - Binary classification for conflict-related content
  - Determines if text is related to conflict, violence, or politics
  - Provides confidence scores for classifications

- **Multi-label Classification**
  - Categorizes text into multiple event types
  - Categories include: Armed Assault, Bombing or Explosion, Kidnapping, and Other
  - Provides confidence scores for each category

- **Question Answering**
  - Extracts answers from provided context based on questions
  - Specialized for conflict-related queries

## Installation

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/conflibert-demo.git](https://github.com/shreyasmeher/conflibert-gui.git)
cd conflibert-demo
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- TensorFlow
- Transformers
- Gradio
- Pandas

## Usage

### Running the Application

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Using Different Features

#### Individual Text Analysis

1. Select the desired task from the dropdown menu:
   - Named Entity Recognition
   - Text Classification
   - Multilabel Classification
   - Question Answering

2. For standard tasks:
   - Enter your text in the input box
   - Click Submit

3. For Question Answering:
   - Enter the context in the context box
   - Enter your question in the question box
   - Click Submit

#### Batch Processing with CSV

1. Prepare a CSV file with a 'text' column containing your texts

2. Select the desired task:
   - NER
   - Text Classification
   - Multilabel Classification

3. Upload your CSV file using the file upload component

4. Click Submit to process the entire file

5. Download the results CSV containing the original text and analysis results

## Model Information

ConfliBERT uses several specialized models:

- **NER Model**: `eventdata-utd/conflibert-named-entity-recognition`
- **Binary Classification**: `eventdata-utd/conflibert-binary-classification`
- **Multi-label Classification**: `eventdata-utd/conflibert-satp-relevant-multilabel`
- **Question Answering**: `salsarra/ConfliBERT-QA`

## Output Formats

### NER Output
```
EntityType: Entity1, Entity2 || EntityType2: Entity3 | Entity4
```

### Binary Classification Output
```
Class (Confidence%)
```

### Multi-label Classification Output
```
Class1 (Confidence%) | Class2 (Confidence%)
```

## Technical Details

### File Structure
```
conflibert-demo/
├── app.py              # Main application file
├── requirements.txt    # Package dependencies
└── README.md          # Documentation
```

### Key Components

- **UI Components**: Built using Gradio
- **Backend Processing**: PyTorch and TensorFlow
- **Data Processing**: Pandas for CSV handling
- **Model Integration**: Hugging Face Transformers

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Credits

Developed by:
- [Sultan Alsarra](https://www.linkedin.com/in/sultan-alsarra-phd-56977a63/)
- [Shreyas Meher](http://shreyasmeher.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Institutional Support

- [UTD Event Data](https://eventdata.utdallas.edu/)
- [University of Texas at Dallas](https://www.utdallas.edu/)

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{conflibert2024,
  title = {ConfliBERT: A BERT-based Model for Conflict and Political Event Analysis},
  author = {Alsarra, Sultan and Meher, Shreyas},
  year = {2024},
  publisher = {UTD Event Data},
  url = {https://eventdata.utdallas.edu/conflibert/}
}
```
