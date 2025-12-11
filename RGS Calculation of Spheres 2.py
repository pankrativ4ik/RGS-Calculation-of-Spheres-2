import os
import sys
import json
import math
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

C = 299792458.0  # скорость света, м/с

def tofloat(v):
    """Преобразует значения из YAML (в т.ч. строки '10e-3') в float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        # безопасный float
        return float(v)
    raise ValueError("Неожиданный тип числа: {}".format(type(v)))

class RCSCalculator:
    """
    Класс, отвечающий за расчет ЭПР (RCS) идеально проводящей сферы.
    """

    def __init__(self, diameter_m):
        self.D = float(diameter_m)
        self.r = self.D / 2.0

    def _num_terms(self, kr):
        """
        Оценка числа членов ряда Mie-подобно.
        Используем классическую оценку: N ~ kr + 4*(kr)^(1/3) + 2, с минимумом 20.
        """
        try:
            n_est = int(math.ceil(kr + 4.0 * (kr ** (1.0 / 3.0)) + 2.0))
        except Exception:
            n_est = 20
        return max(20, n_est)

    def compute_rcs_for_frequency(self, freq_hz):
        """
        Считает ЭПР для одного значения частоты (в Гц).
        Возвращает RCS в м^2.
        """
        wavelength = C / freq_hz
        k = 2.0 * math.pi / wavelength
        kr = k * self.r

        # если kr == 0 (теоретически) — RCS = 0
        if kr <= 0:
            return 0.0

        N = self._num_terms(kr)

        # сумма S = sum_{n=1..N} (-1)^n (n+0.5) (b_n - a_n)
        S = 0+0j
        for n in range(1, N + 1):
            # j_n(kr) and y_n(kr)
            jn = special.spherical_jn(n, kr)
            yn = special.spherical_yn(n, kr)
            hn = jn + 1j * yn  # spherical Hankel

            # j_{n-1}(kr) and h_{n-1}(kr)
            jn1 = special.spherical_jn(n - 1, kr)
            yn1 = special.spherical_yn(n - 1, kr)
            hn1 = jn1 + 1j * yn1

            # an = jn / hn
            # guard against division by zero
            if hn == 0:
                an = 0+0j
            else:
                an = jn / hn

            # numerator and denominator for b_n
            num = (kr * jn1) - (n * jn)
            den = (kr * hn1) - (n * hn)
            if den == 0:
                bn = 0+0j
            else:
                bn = num / den

            S += ((-1) ** n) * (n + 0.5) * (bn - an)

        sigma = (wavelength ** 2 / math.pi) * (abs(S) ** 2)
        return float(sigma)

    def compute_rcs_array(self, freqs_hz):
        """Векторно: принимает iterable частот в Гц, возвращает (lambda_m_array, rcs_array)."""
        freqs = np.asarray(freqs_hz, dtype=float)
        lambdas = C / freqs
        rcs_list = []
        for f in freqs:
            rcs_list.append(self.compute_rcs_for_frequency(float(f)))
        return lambdas.tolist(), rcs_list

class ResultWriter:
    """Класс, отвечающий за вывод/сохранение результатов."""

    @staticmethod
    def save_json(freqs_hz, lambdas_m, rcs_list, out_filename):
        out = {
            "freq": [float(f) for f in freqs_hz],
            "lambda": [float(l) for l in lambdas_m],
            "rcs": [float(r) for r in rcs_list],
        }
        with open(out_filename, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        return out_filename

def load_variants_from_yaml(yaml_path):
    """Читает YAML файл и возвращает словарь вариантов."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    # Ожидается структура { 'data': { '1': {D:..., fmin:..., fmax:...}, ... } }
    data = content.get("data", content)
    # преобразуем ключи в int и значения в словари с float
    variants = {}
    for k, val in data.items():
        try:
            ik = int(k)
        except Exception:
            ik = int(k)
        # D,fmin,fmax могут быть строками типа '10e-3'
        D = tofloat(val["D"])
        fmin = tofloat(val["fmin"])
        fmax = tofloat(val["fmax"])
        variants[ik] = {"D": D, "fmin": fmin, "fmax": fmax}
    return variants

def main():
    yaml_filename = "task_rcs_02.yaml"
    if not os.path.exists(yaml_filename):
        print(f"Файл параметров '{yaml_filename}' не найден в текущей папке: {os.getcwd()}")
        sys.exit(1)

    variants = load_variants_from_yaml(yaml_filename)

    print("Найденные варианты (из файла {}):".format(yaml_filename))
    for k in sorted(variants.keys()):
        v = variants[k]
        print(f"  {k:2d} : D = {v['D']:.6g} m, fmin = {v['fmin']:.6g} Hz, fmax = {v['fmax']:.6g} Hz")

    # Запрос варианта у пользователя
    default_variant = 11
    sel = input(f"\nВыберите номер варианта для расчёта: ").strip()
    if sel == "":
        variant_num = default_variant
    else:
        try:
            variant_num = int(sel)
        except:
            print("Неверный ввод — используем вариант по умолчанию:", default_variant)
            variant_num = default_variant

    if variant_num not in variants:
        print(f"Вариант {variant_num} не найден в файле. Доступные: {sorted(variants.keys())}")
        sys.exit(1)

    params = variants[variant_num]
    D = params["D"]
    fmin = params["fmin"]
    fmax = params["fmax"]

    print(f"\nЗапущен расчёт для варианта {variant_num}: D={D} m, fmin={fmin} Hz, fmax={fmax} Hz")

    # частоты: используем линейную сетку из 400 точек (можно изменить)
    Npoints = 400
    freqs = np.linspace(fmin, fmax, Npoints)

    calc = RCSCalculator(D)
    lambdas, rcs = calc.compute_rcs_array(freqs)

    # сохранить JSON (формат 3)
    out_json = f"rcs_variant{variant_num}.json"
    ResultWriter.save_json(freqs, lambdas, rcs, out_json)
    print(f"\nРезультаты сохранены в файле: {out_json}")

    # вывести первые 20 строк (точек)
    print("\nПервые 20 точек (freq [Hz], lambda [m], rcs [m^2]):")
    for i in range(min(20, len(freqs))):
        print(f"{i+1:3d}: {freqs[i]:.6e} , {lambdas[i]:.6e} , {rcs[i]:.6e}")

    # построить график
    plt.figure(figsize=(8, 5))
    plt.semilogy(freqs / 1e9, rcs)  # частота в ГГц
    plt.xlabel("Частота, ГГц")  # ось X
    plt.ylabel("RCS, м²")
    plt.title(f"RCS vs Frequency — variant {variant_num}, D={D} m")
    plt.grid(True, which="both", ls="--", lw=0.5)
    figname = f"rcs_variant{variant_num}.png"
    plt.tight_layout()
    plt.savefig(figname, dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
