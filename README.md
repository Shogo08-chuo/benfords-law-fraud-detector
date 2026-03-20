# Benford's Law Fraud Detector

Gemini APIによるAI解析を統合した、統計的データ監査補助ツール

## Overview
本プロジェクトは、財務・統計データに対してベンフォードの法則を適用し、異常値の検出とその解釈を支援するデータ監査補助ツールである。  
従来のベンフォード分析は「分布の乖離を確認する」に留まることが多いが、本システムでは統計的乖離の定量化に加え、AIを用いて不正リスクの仮説生成までを一貫して行う。

## Problem
データ監査において以下の課題が存在する：

- ベンフォード分析は「異常の検知」はできるが、解釈が属人的
- 統計結果から具体的な不正リスクへの落とし込みに時間がかかる
- 初学者にとって分析結果の意味理解が難しい

## Solution
本システムは以下の機能により課題を解決する：

1. CSVデータから第1桁を抽出し、理論分布と比較  
2. χ²検定やMADなどにより統計的乖離を定量化  
3. Gemini APIを用いて、乖離結果に基づく不正リスクの仮説を自動生成  

## Key Features

### Benford Analysis
第1桁の出現頻度を理論値と比較し、分布の可視化を行う  

### Statistical Scoring
χ²検定・MAD指標により、データの不自然さを数値化  

### AI-driven Interpretation
Gemini APIを用いて、  
- 想定される不正パターン  
- 監査上の注意点  
を自然言語で出力  

## Tech Stack
- Language: Python  
- Framework: Streamlit  
- Library: Pandas, NumPy, SciPy, Matplotlib  
- AI: Gemini API  

## Limitations
本手法には以下の制約がある：

- ベンフォードの法則は特定条件（桁スケールの広さなど）でのみ有効  
- 小規模データでは信頼性が低下する  
- 本ツールは不正を「証明」するものではなく、あくまで異常検知の補助である  

## Future Work
- 多桁分析（第2桁・桁組み合わせ）への拡張  
- 時系列異常検知との統合  
- 実データを用いた検証強化  

## Author
中央大学 国際情報学部  
剣道歴16年（インターハイ個人・団体出場）  
保有資格: AWS Cloud Practitioner, G検定
