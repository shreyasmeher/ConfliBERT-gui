# ConfliBERT GUI Application

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

### Requirements

**Required:**
- Python 3.8+
- Git
- Code editor (VS Code recommended)

**Optional but recommended:**
- PowerShell 5.0+ (Windows)
- Terminal (Mac)

### Installation Steps

1. Install Python:
   - Download from [python.org](https://www.python.org/downloads/)
   - Check installation: `python --version`

2. Install Git:
   - Windows: Download from [git-scm.com](https://git-scm.com/downloads)
   - Mac: `brew install git` or download from [git-scm.com](https://git-scm.com/downloads)
   - Check installation: `git --version`

3. Clone repository:
```bash
git clone https://github.com/shreyasmeyer/conflibert-gui.git
cd conflibert-gui
```

4. Create and activate virtual environment:
```bash
# Create environment
python -m venv env

# Activate environment
# On Windows:
env\Scripts\activate
# On Mac/Linux:
source env/bin/activate
```

5. For Windows users with permission errors, run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

6. Install requirements:
```bash
pip install -r requirements.txt
```

### Package Requirements

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
conflibert-gui/
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
@inproceedings{hu2022conflibert,
  title={ConfliBERT: A Pre-trained Language Model for Political Conflict and Violence},
  author={Hu, Yibo and Hosseini, MohammadSaleh and Parolin, Erick Skorupa and Osorio, Javier and Khan, Latifur and Brandt, Patrick and D’Orazio, Vito},
  booktitle={Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={5469--5482},
  year={2022}
}
```
