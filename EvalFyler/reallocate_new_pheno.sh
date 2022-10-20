#!/usr/bin/env bash
for split in train test;
do
	for exp in cyanosis PH_1 PH_2 AA;
	do
		python -u ehr2vec.py reallocate ~/downstream/datasets/Fontan/${split}_npy ~/downstream/datasets/${exp}/${split};
	done;
done

