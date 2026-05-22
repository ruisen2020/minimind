"""
生成 eval_pyq.py 的小测试样例 CSV：pyq_sample.csv
- 12 条朋友圈，tab 分隔（id\ttext\tlabel），无表头
- 覆盖明显广告 / 明显非广告 / 边界样例，方便快速 debug

使用：
    python gen_pyq_sample.py
    python eval_pyq.py --weight full_sft --csv_path ./pyq_sample.csv --sample_n 0 --log_every 1
"""
import os

SAMPLES = [
    # id,   text,                                                                label(1=广告 / 0=非广告)
    ("10001", "Dior新款包包到货❗️秋冬经典款 💰1550/4300 有需要私我",                                                     1),
    ("10002", "今天天气真好，带娃去公园散步，小朋友玩得特别开心[太阳]",                                                    0),
    ("10003", "【拉丁语级别1班】本周六开课！导师哈佛牛津教育证书，10次课，名额有限欢迎私信报名！",                         1),
    ("10004", "人到中年，才发现身体健康才是最大的财富，愿大家都平安健康[合十]",                                            0),
    ("10005", "🏡房屋出租：两室一厅全家具家电，走路5分钟到超市，月租$1800，有意联系713-298-0156",                          1),
    ("10006", "刚看完电影《泳者之心》，真的被感动到了，女性力量太伟大[哭泣][哭泣]",                                        0),
    ("10007", "兰芝新款隔离 30ml 💰148包邮，紫色适合暗黄皮肤，绿色适合红血丝，修饰肤色超好用",                              1),
    ("10008", "今天加班到凌晨，一杯咖啡续命ing，老板啥时候给我涨工资啊[捂脸]",                                              0),
    ("10009", "moncler女士羽绒服brou款 1183×0.9=1064.7🉐 需要的宝宝抓紧",                                                   1),
    ("10010", "朋友推荐的那本《当下的力量》真的不错，读完整个人都安静了下来，推荐给大家",                                   0),
    ("10011", "急招data analyst，hybrid在VA办公室，提供sponsorship，要求会sql和tableau，有意联系",                          1),
    ("10012", "外州度假⛱️模式开启🔛！手机正常操作，发信息保持良好心情度假工作两不误[666]",                                  0),
]

OUT_PATH = os.path.join(os.path.dirname(__file__), "pyq_sample.csv")


def main():
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        for sid, text, label in SAMPLES:
            # 把 text 里可能出现的 tab/换行替换成空格，避免破坏分隔
            safe_text = text.replace("\t", " ").replace("\n", " ")
            f.write(f"{sid}\t{safe_text}\t{label}\n")
    print(f"✅ 已生成 {OUT_PATH}")
    print(f"   样本数：{len(SAMPLES)}")
    print(f"   label=1 广告：{sum(1 for _, _, l in SAMPLES if l == 1)} 条")
    print(f"   label=0 非广告：{sum(1 for _, _, l in SAMPLES if l == 0)} 条")

    # 校验：用 eval_pyq.py 里的相同解析逻辑试读一遍
    print("\n🔍 解析校验（模拟 eval_pyq.py 读取）：")
    import csv as _csv
    with open(OUT_PATH, "r", encoding="utf-8", newline="") as f:
        reader = _csv.reader(f, delimiter="\t", quoting=_csv.QUOTE_MINIMAL)
        for i, row in enumerate(reader, 1):
            if len(row) < 3 or row[-1] not in ("0", "1"):
                print(f"  ❌ 第{i}行解析异常：{row}")
            else:
                text_preview = row[1][:30] + ("..." if len(row[1]) > 30 else "")
                print(f"  ✓ id={row[0]:<6} label={row[-1]}  text={text_preview}")


if __name__ == "__main__":
    main()
