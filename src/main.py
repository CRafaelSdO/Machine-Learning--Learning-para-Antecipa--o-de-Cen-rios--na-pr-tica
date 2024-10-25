import pandas as pd

from numpy.random import RandomState, MT19937, SeedSequence
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def main():
    # Dados para análise
    df = pd.read_csv(__file__.replace("src/main.py", "db/wine_dataset.csv"))

    # Modelo
    gnb = GaussianNB()

    # Gerador pseudo aleatório
    rs = RandomState(MT19937(SeedSequence()))

    # Separa o target (y) das características (x) dos dados
    df_y = df["style"]
    df_x = df.drop("style", axis=1)

    # Separa os grupos de treinamento e teste
    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3)

    # Treina o modelo
    gnb.fit(train_x, train_y)

    # Calcula a acurácia do modelo
    accuracy = gnb.score(test_x, test_y)

    # Cria uma amostra aleatória
    sample = df.sample(9, random_state=rs)

    # Separa o target (y) das características (x) da amostra
    sample_y = sample["style"]
    sample_x = sample.drop("style", axis=1)

    # Faz uma predição aplicando o modelo às características da amostra
    sample_predict = gnb.predict(sample_x)

    # Mostra a acurácia do modelo
    print(f"Acurácia: {accuracy:.2f}")

    # Mostra a predição e o resultado real lado a lado
    print(f"Predição: {sample_predict}")
    print(f"Real:     {sample_y.values}")


if __name__ == "__main__":
    main()
