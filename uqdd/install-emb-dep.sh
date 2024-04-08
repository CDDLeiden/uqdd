#!/bin/bash

replace_content() {
    local source_file="$1"
    local target_file="$2"

    if [[ ! -f "$source_file" ]]; then
        printf "Source file does not exist: %s\n" "$source_file" >&2
        return 1
    fi

    if [[ ! -f "$target_file" ]]; then
        printf "Target file does not exist: %s\n" "$target_file" >&2
        return 1
    fi

    if ! cp -f "$source_file" "$target_file"; then
        printf "Failed to replace the content of %s with %s\n" "$target_file" "$source_file" >&2
        return 1
    fi

    printf "Successfully replaced the content of %s with %s\n" "$target_file" "$source_file"
}

main() {
    local source_file="requirements-biotransformer.txt"
    local target_file="bio-transformers-0.1.17/requirements.txt"

    replace_content "$source_file" "$target_file"
}

wget https://files.pythonhosted.org/packages/34/5a/086fd5c388c4a6c0fb5f0e541b6ec922187a09d85f829789a4ff72f34295/bio-transformers-0.1.17.tar.gz
tar -xvf bio-transformers-0.1.17.tar.gz
main
rm -rf bio-transformers-0.1.17.tar.gz
pip install -r requirements-biotransformer.txt

cd bio-transformers-0.1.17 || exit
pip install .
cd ..
# remove the directory
rm -rf ./bio-transformers-0.1.17
