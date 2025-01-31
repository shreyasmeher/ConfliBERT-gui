import torch
import tensorflow as tf
from tf_keras import models, layers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, TFAutoModelForQuestionAnswering
import gradio as gr
import re
import pandas as pd
import io

# Check if GPU is available and use it if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_TOKEN_LENGTH = 512  # Adjust based on your model's limits

def truncate_text(text, tokenizer, max_length=MAX_TOKEN_LENGTH):
    """Truncate text to max token length"""
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length-1] + [tokenizer.sep_token_id]
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return text

def safe_process(func, text, tokenizer):
    """Safely process text with proper error handling"""
    try:
        truncated_text = truncate_text(text, tokenizer)
        return func(truncated_text)
    except Exception as e:
        error_msg = str(e)
        if 'out of memory' in error_msg.lower():
            return "Error: Text too long for processing"
        elif 'cuda' in error_msg.lower():
            return "Error: GPU processing error"
        else:
            return f"Error: {error_msg}"

# Load the models and tokenizers
qa_model_name = 'salsarra/ConfliBERT-QA'
qa_model = TFAutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

ner_model_name = 'eventdata-utd/conflibert-named-entity-recognition'
ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name).to(device)
ner_tokenizer = AutoTokenizer.from_pretrained(ner_model_name)

clf_model_name = 'eventdata-utd/conflibert-binary-classification'
clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_name).to(device)
clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_name)

multi_clf_model_name = 'eventdata-utd/conflibert-satp-relevant-multilabel'
multi_clf_model = AutoModelForSequenceClassification.from_pretrained(multi_clf_model_name).to(device)
multi_clf_tokenizer = AutoTokenizer.from_pretrained(multi_clf_model_name)

# Define the class names for text classification
class_names = ['Negative', 'Positive']
multi_class_names = ["Armed Assault", "Bombing or Explosion", "Kidnapping", "Other"]  # Updated labels

# Define the NER labels and colors
ner_labels = {
    'Organisation': 'blue',
    'Person': 'red',
    'Location': 'green',
    'Quantity': 'orange',
    'Weapon': 'purple',
    'Nationality': 'cyan',
    'Temporal': 'magenta',
    'DocumentReference': 'brown',
    'MilitaryPlatform': 'yellow',
    'Money': 'pink'
}

def handle_error_message(e, default_limit=512):
    error_message = str(e)
    pattern = re.compile(r"The size of tensor a \((\d+)\) must match the size of tensor b \((\d+)\)")
    match = pattern.search(error_message)
    if match:
        number_1, number_2 = match.groups()
        return f"<span style='color: red; font-weight: bold;'>Error: Text Input is over limit where inserted text size {number_1} is larger than model limits of {number_2}</span>"
    pattern_qa = re.compile(r"indices\[0,(\d+)\] = \d+ is not in \[0, (\d+)\)")
    match_qa = pattern_qa.search(error_message)
    if match_qa:
        number_1, number_2 = match_qa.groups()
        return f"<span style='color: red; font-weight: bold;'>Error: Text Input is over limit where inserted text size {number_1} is larger than model limits of {number_2}</span>"
    return f"<span style='color: red; font-weight: bold;'>Error: Text Input is over limit where inserted text size is larger than model limits of {default_limit}</span>"

# Define the functions for each task
def question_answering(context, question):
    try:
        inputs = qa_tokenizer(question, context, return_tensors='tf', truncation=True)
        outputs = qa_model(inputs)
        answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
        answer_end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1
        answer = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'].numpy()[0][answer_start:answer_end]))
        return f"<span style='color: green; font-weight: bold;'>{answer}</span>"
    except Exception as e:
        return handle_error_message(e)

def replace_unk(tokens):
    return [token.replace('[UNK]', "'") for token in tokens]

def named_entity_recognition(text, output_format='html'):
    """
    Process text for named entity recognition.
    output_format: 'html' for GUI display, 'csv' for CSV processing
    """
    try:
        inputs = ner_tokenizer(text, return_tensors='pt', truncation=True)
        with torch.no_grad():
            outputs = ner_model(**inputs)
        ner_results = outputs.logits.argmax(dim=2).squeeze().tolist()
        tokens = ner_tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        tokens = replace_unk(tokens)
        
        entities = []
        seen_labels = set()
        current_entity = []
        current_label = None
        
        # Process tokens and group consecutive entities
        for i in range(len(tokens)):
            token = tokens[i]
            label = ner_model.config.id2label[ner_results[i]].split('-')[-1]
            
            # Handle subwords
            if token.startswith('##'):
                if entities:
                    if output_format == 'html':
                        entities[-1][0] += token[2:]
                    elif current_entity:
                        current_entity[-1] = current_entity[-1] + token[2:]
            else:
                # For CSV format, group consecutive tokens of same entity type
                if output_format == 'csv':
                    if label != 'O':
                        if label == current_label:
                            current_entity.append(token)
                        else:
                            if current_entity:
                                entities.append([' '.join(current_entity), current_label])
                            current_entity = [token]
                            current_label = label
                    else:
                        if current_entity:
                            entities.append([' '.join(current_entity), current_label])
                            current_entity = []
                            current_label = None
                else:
                    entities.append([token, label])
                
                if label != 'O':
                    seen_labels.add(label)
        
        # Don't forget the last entity for CSV format
        if output_format == 'csv' and current_entity:
            entities.append([' '.join(current_entity), current_label])
        
        if output_format == 'csv':
            # Group by entity type
            grouped_entities = {}
            for token, label in entities:
                if label != 'O':
                    if label not in grouped_entities:
                        grouped_entities[label] = []
                    grouped_entities[label].append(token)
            
            # Format the output
            result_parts = []
            for label, tokens in grouped_entities.items():
                unique_tokens = list(dict.fromkeys(tokens))  # Remove duplicates
                result_parts.append(f"{label}: {' | '.join(unique_tokens)}")
            
            return ' || '.join(result_parts)
        else:
            # Original HTML output
            highlighted_text = ""
            for token, label in entities:
                color = ner_labels.get(label, 'black')
                if label != 'O':
                    highlighted_text += f"<span style='color: {color}; font-weight: bold;'>{token}</span> "
                else:
                    highlighted_text += f"{token} "

            legend = "<div><strong>NER Tags Found:</strong><ul style='list-style-type: disc; padding-left: 20px;'>"
            for label in seen_labels:
                color = ner_labels.get(label, 'black')
                legend += f"<li style='color: {color}; font-weight: bold;'>{label}</li>"
            legend += "</ul></div>"
            
            return f"<div>{highlighted_text}</div>{legend}"
            
    except Exception as e:
        return handle_error_message(e)

def text_classification(text):
    try:
        inputs = clf_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = clf_model(**inputs)
        logits = outputs.logits.squeeze().tolist()
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item() * 100

        if predicted_class == 1:  # Positive class
            result = f"<span style='color: green; font-weight: bold;'>Positive: The text is related to conflict, violence, or politics. (Confidence: {confidence:.2f}%)</span>"
        else:  # Negative class
            result = f"<span style='color: red; font-weight: bold;'>Negative: The text is not related to conflict, violence, or politics. (Confidence: {confidence:.2f}%)</span>"
        return result
    except Exception as e:
        return handle_error_message(e)

def multilabel_classification(text):
    try:
        inputs = multi_clf_tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = multi_clf_model(**inputs)
        predicted_classes = torch.sigmoid(outputs.logits).squeeze().tolist()
        if len(predicted_classes) != len(multi_class_names):
            return f"Error: Number of predicted classes ({len(predicted_classes)}) does not match number of class names ({len(multi_class_names)})."

        results = []
        for i in range(len(predicted_classes)):
            confidence = predicted_classes[i] * 100
            if predicted_classes[i] >= 0.5:
                results.append(f"<span style='color: green; font-weight: bold;'>{multi_class_names[i]} (Confidence: {confidence:.2f}%)</span>")
            else:
                results.append(f"<span style='color: red; font-weight: bold;'>{multi_class_names[i]} (Confidence: {confidence:.2f}%)</span>")

        return " / ".join(results)
    except Exception as e:
        return handle_error_message(e)
    
def clean_html_tags(text):
    """Remove HTML tags and formatting from the output."""
    # Remove HTML tags but keep the text content
    clean_text = re.sub(r'<[^>]+>', '', text)
    # Remove multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # Remove [CLS] and [SEP] tokens
    clean_text = re.sub(r'\[CLS\]|\[SEP\]', '', clean_text)
    return clean_text.strip()

def extract_ner_entities(html_output):
    """Extract entities and their types from NER output using a simpler approach."""
    # Map colors to entity types
    color_to_type = {
        'blue': 'Organisation',
        'red': 'Person',
        'green': 'Location',
        'orange': 'Quantity',
        'purple': 'Weapon',
        'cyan': 'Nationality',
        'magenta': 'Temporal',
        'brown': 'DocumentReference',
        'yellow': 'MilitaryPlatform',
        'pink': 'Money'
    }
    
    # Find all colored spans
    pattern = r"<span style='color: ([^']+)[^>]+>([^<]+)</span>"
    matches = re.findall(pattern, html_output)
    
    # Group by entity type
    entities = {}
    
    # Process each match
    for color, text in matches:
        if color in color_to_type:
            entity_type = color_to_type[color]
            if entity_type not in entities:
                entities[entity_type] = []
            
            # Clean and store the text
            text = text.strip()
            if text and not text.isspace():
                entities[entity_type].append(text)
    
    # Join consecutive words for each entity type
    result_parts = []
    for entity_type, words in entities.items():
        # Join consecutive words
        phrases = []
        current_phrase = []
        
        for word in words:
            if word in [',', '/', ':', '-']:  # Skip punctuation
                continue
            if not current_phrase:
                current_phrase.append(word)
            else:
                # If it's a continuation (e.g., part of a date or name)
                if word.startswith(':') or word == 'of' or current_phrase[-1].endswith('/'):
                    current_phrase.append(word)
                else:
                    # If it's a new entity
                    phrases.append(' '.join(current_phrase))
                    current_phrase = [word]
        
        if current_phrase:
            phrases.append(' '.join(current_phrase))
        
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in phrases:
            clean_phrase = phrase.strip()
            if clean_phrase and clean_phrase not in seen:
                unique_phrases.append(clean_phrase)
                seen.add(clean_phrase)
        
        if unique_phrases:
            result_parts.append(f"{entity_type}: {' | '.join(unique_phrases)}")
    
    return ' || '.join(result_parts)


def clean_classification_output(html_output):
    """Extract classification results without HTML formatting."""
    if "Positive" in html_output:
        # Binary classification
        match = re.search(r">(Positive|Negative).*?Confidence: ([\d.]+)%", html_output)
        if match:
            class_name, confidence = match.groups()
            return f"{class_name} ({confidence}%)"
    else:
        # Multilabel classification
        results = []
        matches = re.finditer(r">([^<]+)\s*\(Confidence:\s*([\d.]+)%\)", html_output)
        for match in matches:
            class_name, confidence = match.groups()
            if float(confidence) >= 50:  # Only include classes with confidence >= 50%
                results.append(f"{class_name.strip()} ({confidence}%)")
        return " | ".join(results) if results else "No classes above 50% confidence"
    
    return "Unknown"

    
def process_csv_ner(file):
    try:
        df = pd.read_csv(file.name)
        
        if 'text' not in df.columns:
            return "Error: CSV must contain a 'text' column"
            
        entities = []
        for text in df['text']:
            if pd.isna(text):
                entities.append("")
                continue
            
            # Use CSV output format
            result = named_entity_recognition(str(text), output_format='csv')
            entities.append(result)
        
        df['entities'] = entities
        
        output_path = "processed_results.csv"
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        return f"Error processing CSV: {str(e)}"
    
def process_csv_classification(file, is_multi=False):
    try:
        df = pd.read_csv(file.name)
        
        if 'text' not in df.columns:
            return "Error: CSV must contain a 'text' column"
            
        results = []
        for text in df['text']:
            if pd.isna(text):
                results.append("")
                continue
                
            if is_multi:
                html_result = multilabel_classification(str(text))
            else:
                html_result = text_classification(str(text))
            results.append(clean_classification_output(html_result))
            
        result_column = 'multilabel_results' if is_multi else 'classification_results'
        df[result_column] = results
        
        output_path = "processed_results.csv"
        df.to_csv(output_path, index=False)
        return output_path
    except Exception as e:
        return f"Error processing CSV: {str(e)}"


# Define the Gradio interface
def chatbot(task, text=None, context=None, question=None, file=None):
    if file is not None:  # Handle CSV file input
        if task == "Named Entity Recognition":
            return process_csv_ner(file)
        elif task == "Text Classification":
            return process_csv_classification(file, is_multi=False)
        elif task == "Multilabel Classification":
            return process_csv_classification(file, is_multi=True)
        else:
            return "CSV processing is not supported for Question Answering task"
    
    # Handle regular text input (previous implementation)
    if task == "Question Answering":
        if context and question:
            return question_answering(context, question)
        else:
            return "Please provide both context and question for the Question Answering task."
    elif task == "Named Entity Recognition":
        if text:
            return named_entity_recognition(text)
        else:
            return "Please provide text for the Named Entity Recognition task."
    elif task == "Text Classification":
        if text:
            return text_classification(text)
        else:
            return "Please provide text for the Text Classification task."
    elif task == "Multilabel Classification":
        if text:
            return multilabel_classification(text)
        else:
            return "Please provide text for the Multilabel Classification task."
    else:
        return "Please select a valid task."


with gr.Blocks(theme="ParityError/Interstellar") as demo:
    with gr.Column():
        with gr.Row(elem_id="header", elem_classes="header-container"):
            gr.Markdown("<div class='header-title-center'><a href='https://eventdata.utdallas.edu/conflibert/' style='font-size: 4rem; font-weight: 900;'>ConfliBERT</a></div>")
        
        with gr.Column(elem_classes="task-container"):
            gr.Markdown("<h2 style='font-size: 1.25rem; font-weight: 600; margin-bottom: 1.5rem;'>Select a task and provide the necessary inputs:</h2>")
            
            task = gr.Dropdown(
                choices=["Question Answering", "Named Entity Recognition", "Text Classification", "Multilabel Classification"],
                label="Select Task",
                value="Named Entity Recognition"
            )
            
            with gr.Row():
                text_input = gr.Textbox(
                    lines=5,
                    placeholder="Enter the text here...",
                    label="Text",
                    elem_classes="input-text"
                )
                context_input = gr.Textbox(
                    lines=5,
                    placeholder="Enter the context here...",
                    label="Context",
                    visible=False,
                    elem_classes="input-text"
                )
                question_input = gr.Textbox(
                    lines=2,
                    placeholder="Enter your question here...",
                    label="Question",
                    visible=False,
                    elem_classes="input-text"
                )
            
            with gr.Row():
                file_input = gr.File(
                    label="Or upload a CSV file (must contain a 'text' column)",
                    file_types=[".csv"],
                    elem_classes="file-upload"
                )
                file_output = gr.File(
                    label="Download processed results",
                    visible=False,
                    elem_classes="file-download"
                )
            
            with gr.Row():
                submit_button = gr.Button(
                    "Submit",
                    elem_id="submit-button",
                    elem_classes="submit-btn"
                )
            
            output = gr.HTML(label="Output", elem_classes="output-html")
    
    with gr.Row(elem_classes="footer"):
        gr.Markdown("<a href='https://eventdata.utdallas.edu/'>UTD Event Data</a> | <a href='https://www.utdallas.edu/'>University of Texas at Dallas</a>")
        gr.Markdown("Developed By: <a href='https://www.linkedin.com/in/sultan-alsarra-phd-56977a63/' target='_blank'>Sultan Alsarra</a> and <a href='http://shreyasmeher.com' target='_blank'>Shreyas Meher</a>")

    def update_inputs(task_name):
        """Updates the visibility of input components based on the selected task."""
        if task_name == "Question Answering":
            return [
                gr.update(visible=False), 
                gr.update(visible=True), 
                gr.update(visible=True), 
                gr.update(visible=False),
                gr.update(visible=False)
            ]
        else:
            return [
                gr.update(visible=True), 
                gr.update(visible=False), 
                gr.update(visible=False), 
                gr.update(visible=True),
                gr.update(visible=True)
            ]

    def chatbot_interface(task, text, context, question, file):
        """Handles both file and text inputs for different tasks."""
        if file:
            result = chatbot(task, file=file)
            if isinstance(result, str) and result.endswith('.csv'):
                return gr.update(visible=False), gr.update(value=result, visible=True)
            return gr.update(value=result, visible=True), gr.update(visible=False)
        else:
            result = chatbot(task, text, context, question)
            return gr.update(value=result, visible=True), gr.update(visible=False)

    def chatbot(task, text=None, context=None, question=None, file=None):
        """Main function to process different types of inputs and tasks."""
        if file is not None:  # Handle CSV file input
            if task == "Named Entity Recognition":
                return process_csv_ner(file)
            elif task == "Text Classification":
                return process_csv_classification(file, is_multi=False)
            elif task == "Multilabel Classification":
                return process_csv_classification(file, is_multi=True)
            else:
                return "CSV processing is not supported for Question Answering task"
        
        # Handle regular text input
        if task == "Question Answering":
            if context and question:
                return question_answering(context, question)
            else:
                return "Please provide both context and question for the Question Answering task."
        elif task == "Named Entity Recognition":
            if text:
                return named_entity_recognition(text)
            else:
                return "Please provide text for the Named Entity Recognition task."
        elif task == "Text Classification":
            if text:
                return text_classification(text)
            else:
                return "Please provide text for the Text Classification task."
        elif task == "Multilabel Classification":
            if text:
                return multilabel_classification(text)
            else:
                return "Please provide text for the Multilabel Classification task."
        else:
            return "Please select a valid task."

    task.change(fn=update_inputs, inputs=task, outputs=[text_input, context_input, question_input, file_input, file_output])
    submit_button.click(
        fn=chatbot_interface,
        inputs=[task, text_input, context_input, question_input, file_input],
        outputs=[output, file_output]
    )

demo.launch(share=True)
