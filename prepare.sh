#!/bin/zsh

for size in {1000,10000,100000}; do
    for lang in {en,es}; do
        gshuf -n $size europarl-v7.es-en.$lang >| $lang.$size.shuffled
        head -n$((size / 10)) $lang.$size.shuffled > $lang.$size.shuffled.test
        tail +$((size / 10 + 1)) $lang.$size.shuffled > $lang.$size.shuffled.train
    done
done
