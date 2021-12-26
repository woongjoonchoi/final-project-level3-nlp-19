# AI Paperboy

## 📋 Project Abstract

### Purpose

* 사용자가 질문으로 요청하면 답변이 있는 뉴스 기사를 스크랩 해주는 서비스

### Functions
* 기사를 읽고 다시 보고 싶은 기사 스크랩 기능
* 실시간 질문에 대한 답변 기사 AI 스크랩 기능
* 일정 시간마다 기사 목록 업데이트 후 질문에 대한 답변 기사 AI 스크랩 기능

## 👨‍👨‍👧‍👦 Team Covit-19

### Members

| <center>박별이</center> | <center>이준수</center> | <center>최웅준</center> | <center>추창한</center>
| -------- | -------- | -------- | -------- |
| [<img src="https://i.imgur.com/1zMaAt1.png" height=100px width=100px></img>](https://github.com/ParkByeolYi) | [<img src="https://i.imgur.com/o3BFRGk.png" height=100px width=100px></img>](https://github.com/JunsooLee) | [<img src="https://i.imgur.com/GzN3ZOv.png" height=100px width=100px></img>](https://github.com/woongjoonchoi) | [<img src="https://i.imgur.com/S4cM768.png" height=100px width=100px></img>](https://github.com/cnckdgks) |
| <center>[github](https://github.com/ParkByeolYi)</center> | <center>[github](https://github.com/JunsooLee)</center> | <center>[github](https://github.com/woongjoonchoi)</center> | <center>[github](https://i.imgur.com/S4cM768.png)</center> |


### Responsibilities
|                     | 박별이 | 이준수 | 최웅준 | 추창한 |
| ------------------- | ------ | ------ | ------ | ------ |
| Code refactoring    | Retrieval | post_processing <br> train |extraction_pre_process <br>generation_pre_process <br>generation_compute_metrics <br>configuration  <br>building tiny dataset  | Retrieval |
| User flow/Data flow |        | User Flow <br> Data Flow |    training pipeline    |  User Flow<br> Data Flow  |
| Modeling            |        | build train dataset <br> model training | train with tiny dataset <br>training reader model <br> error analysis on generation model         |   Apply BM 25   |
| Prototyping         |        |        |  reader model demo      |  ODQA model / Batch Serving      |
| Frontend            |  sign in <br> sign up <br> news scrap  | article_form |  homepage_news title list <br> ai scrap news title list <br>my scrap news title list   |        |
| Backend             | sign in <br> sign up <br> news scrap | user_input |  homepage_news title list  with wiki_news_db<br> ai scrap news title list  with ai_scrap_db<br>my scrap news title list  with user_scrap_db     |  get article page and user_input with real time service <br> batch serving |


### Collaboration tool
<img src="https://img.shields.io/badge/Google Drive-4285F4?style=flat-square&logo=Google Drive&logoColor=white"/> <img src="https://img.shields.io/badge/MS ToDo-6264A7?style=flat-square&logo=Microsoft&logoColor=white"/> <img src="https://img.shields.io/badge/Notion-5E5E5E?style=flat-square&logo=Notion&logoColor=white"/> 

## 💾 Installation
### 1. Set up the python environment:
- Recommended python version 3.8.5

```
$ conda create -n venv python=3.8.5 pip
$ conda activate venv
```
### 2. Install other required packages

```
$ cd $ROOT/final-project-level3-nlp-19/code
$ poetry install
$ poetry shell
```

## 🖥 Usage
### 1. Project Structure
```
code
├──routers/
├──schema/
├──services/
├──templates/
├──AIPaperboy.py
└──model train file (.py)
```
**4 folder for serving**
- **routers**: Controller
- **schema**: Model
- **sevices**: Project's functions
- **templates**: HTML & CSS file

### 2. Execute
```
$ cd $ROOT/final-project-level3-nlp-19/code
$ python AIPaperboy.py --output_dir ./outputs/test_dataset/ --model_name_or_path ./models/train_dataset/ --dataset_name ../data/test_dataset/ --do_predict
```
## 📽 Demo
* [AI Paperboy Demo Video](https://www.youtube.com/watch?v=n7oPu7vrQ8s)
* [AI Paperboy Presentation](https://docs.google.com/presentation/d/1rpgp9knamiiqs4lITZMEiixSA8sfWyvv/edit?usp=sharing&ouid=110643334622897859461&rtpof=true&sd=true)
