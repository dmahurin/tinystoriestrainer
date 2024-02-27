import argparse
import os
from transformers import GPTNeoConfig, GPTNeoForCausalLM, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from tqdm import tqdm
from transformers import AutoTokenizer
import settings

reuse_tokenizer = os.path.exists(settings.OUTPUT_DIR + '/' + 'tokenizer.model')
tokenizer_path = settings.OUTPUT_DIR if reuse_tokenizer else settings.tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Define the features of your dataset
"""
features = Features({
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(feature=Value(dtype='int32')),
    # Define other features if you have them
})
"""


def load_stories(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        story = []
        for line in f:
            line = line.strip()
            if line == settings.END_OF_TEXT:
                yield '\n'.join(story)
                story = []
            else:
                story.append(line)

        if story:  # handle last story in file
            yield '\n'.join(story)


def create_dataset(stories: list, padding_option):
    dataset = Dataset.from_list([{'text': text} for text in stories])
    dataset = dataset.map(lambda examples: tokenizer(examples['text'], truncation=True, padding=padding_option, max_length=settings.MAX_LENGTH), batched=True, remove_columns=["text"], num_proc=8)
    # dataset.cast(features)
    return dataset


def load_data(train_path, valid_path, padding_option):
    train = list(tqdm(load_stories(train_path), desc="Loading Training data"))
    if valid_path is None:
        valid_size = -(-len(train)//settings.VALIDATION_FRACTION)
        valid = train[:valid_size]
        train = train[valid_size:]
    else:
        valid = list(tqdm(load_stories(valid_path), desc="Loading Validation data"))
    return DatasetDict({
        'test': create_dataset(valid, padding_option),
        'train': create_dataset(train, padding_option)
    })


def main(padding_option):
    if padding_option != 'do_not_pad':
        tokenizer.pad_token = tokenizer.eos_token

    data = load_data(settings.TRAIN_PATH, settings.VALID_PATH, padding_option)
    data.save_to_disk("prepared_tinystories2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", help="padding option: 'max_length', 'longest', 'do_not_pad'", default='do_not_pad')
    args = parser.parse_args()

    main(args.padding)
