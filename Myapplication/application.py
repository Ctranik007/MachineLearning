import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
import streamlit as st
from catboost import CatBoostRegressor

def show_title():
    # Заголовок и подзаголовок
    st.title("Прогнозирование продаж")
    st.write("# Методы регрессии")

def show_info():
    st.write("### Задача")
    st.write(
        "Построить, обучить и оценить модель для решения задачи регрессии - спрогнозировать общий объем продаж для каждого продукта и магазина в следующем месяце.")
    st.write("### Описание входных данных")
    st.write(
        "Данные, для которых необходимо получать предсказания, представляют собой подробное признаковое описание продаж в предыдущих месяцев,уникальный идентификатор магазина,уникальный идентификатор продукта")
    st.write("### Выбранная регрессионная модель")
    st.write(
        "В результате анализа метрик качества нескольких продвинутых регрессионных композиционных моделей выбрана модель"
        "RandomForestRegressor, обеспечивающая более высокое качество предсказаний продаж.")
    st.write("Выполненная работа представляет собой результат участия в соревновании на портале Kaggle. Более подробно"
             "ознакомиться с правилами соревнования можно по ссылке ниже:")
    st.write("https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview")

def show_predictions():
    st.write("Файл для примера: https://drive.google.com/file/d/1Kx-e994oXolzfl5CQRtcRommd1YEDaMN/view?usp=sharing")
    file = st.file_uploader(label="Выберите csv файл с данными для прогнозирования количества продаж",
                            type=["csv"],
                            accept_multiple_files=False)
    if file is not None:
        test_data = pd.read_csv(file)
        st.write("### Загруженные данные")
        st.write(test_data)
        make_prediction(get_model(), test_data)

def get_model():
    return CatBoostRegressor().load_model(os.path.join(os.path.dirname(__file__), "Model", "catboots"))


def make_prediction(model, X):
    st.write("### Предсказанные значения")
    pred = pd.DataFrame(model.predict(X))
    st.write(pred)
    st.write("### Гистограмма распределения предсказаний")
    plot_hist(pred)


def plot_hist(data):
    fig = plt.figure()
    sbn.histplot(data, legend=False)
    st.pyplot(fig)


def select_page():
    return st.sidebar.selectbox("Выберите страницу", ("Информация", "Предсказание"))

show_title()
st.sidebar.title("Меню")
page = select_page()
st.sidebar.write("DZHALIL DYUSEKENOV 2021")


if page == "Информация":
    show_info()
else:
    show_predictions()










