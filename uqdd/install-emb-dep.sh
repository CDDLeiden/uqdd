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

main