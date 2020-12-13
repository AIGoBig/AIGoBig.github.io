---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #fff
header: 'Tokyo Jazug Night 28.'
footer: 'Takekazu Omi @Baleen.Studio'
headingDivider: 1
inlineSVG: true
style: |
    section.right * , h6{
        text-align: right;
        overflow-wrap: normal;
    }
---

<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@700&display=swap');
section {
    font-family: 'Noto Sans JP', sans-serif;
}
</style>

# Introduction of Azure Docker Integration
<!-- _class: right -->

![bg left:40%](https://live.staticflickr.com/65535/49975413353_e8861becb4_h.jpg)

[第28回 Tokyo Jazug Night](https://jazug.connpass.com/event/181811/)

###### by Takekazu Omi(*@Baleen.Studio*)
###### 2020/7/30

# **自己紹介**

![bg auto](./media/1x1.png)
![bg auto](./media/1x1.png)
![bg auto](https://www.baleen.studio/img/company/img-symbol.png)

- 近江 武一
- 最近の活動
  - 会社錬成 [baleen.studio](https://baleen.studio)
    - HPは、Azure Static Web Apps + Cloudflare
  - GitHub
    - Azurite で、Table APIサポートを作り始めたので[PR](https://github.com/Azure/Azurite/pull/522)
    - 自作の [posh-direnv](https://github.com/takekazuomi/posh-direnv) へのPRを merge
    - [北極送り](https://twitter.com/takekazuomi/status/1285367356735098880) になりました

# **はじめに**

最近、Dockerを使うことが多い。コンテナで動くようにしておくと、環境再現性が高く開発効率があがる。6/25に、Docker cli のAzure統合がベータリリース :smile: したので今日は簡単に紹介。

- Dockerの紹介
- Azure Container Instances (以下 ACI)の紹介
- Azure Docker Integration
- **DEMO**

# **今日しない話**

<style scoped>
p {
    vertical-align: middle;
    text-align: center;
    font-weight: bold;
    font-size: 3em;
    color: #366ee4;
}
</style>

kubernetes :blush:

# **Docker紹介**

- 2013年 PyCon [The future of Linux Containers](https://www.youtube.com/watch?v=wW9CAH9nSLs)でお披露目
- アプリケーション、依存関係、任意の Linux/Windows を仮想コンテナーにパッケージ化
- パッケージ化したものを、様々な場所で動かせる。Local, Cloud etc

## 依存関係もパッケージ化されるので、開発、実行の両方で**環境再現性**が高くなる

今回のACIも動作環境の一つ

# **No more dll hell**

モジュール化 再利用と [依存関係地獄](https://en.wikipedia.org/wiki/DLL_Hell)は竹馬の友。環境毎に対策されているが、複雑で壊れやすい。

- java maven
- .NET nuget
- python pip
- node npm

## 繰り返す依存関係地獄に終止符を :+1:

# **雑感（１）**

- OSと依存関係を全部まとめてパッケージにするのは良いアイデア
- パッケージのサイズが重要、小さい方が扱いやすい
- Windowsだと、OSだけでギガバイトクラスになるので厳しい

## なんでも、Dockerに詰め込んでしまえば良いんじゃない :smile: :grin:

---

![bg auto](https://live.staticflickr.com/65535/49997478968_6bdfb96cb9_h.jpg)

# **Azure Container Instances(ACI)**

- Docker Engineが入ったVMをPoolから貸出し
- VM内に複数のコンテナを起動できる
- 2018/4/25 [ACI GA](https://azure.microsoft.com/en-us/blog/azure-container-instances-now-generally-available/)

![bg right h:450px](https://docs.microsoft.com/ja-jp/azure/container-instances/media/container-instances-container-groups/container-groups-example.png)

# **ACI 特徴（１）**

- 高速スタートアップ:
  - プロビジョニングに数秒（by 公式ドキュメント)
- コンテナー アクセス
  - コンテナを public endpoint でインターネットに直接公開可
- ハイパーバイザーレベルのセキュリティ
  - コンテナーの分離レベルは、Hypervisor-level。マルチテナント利用でも安全

# **ACI 特徴（２）**

- カスタム サイズ
  - CPU のコア数とメモリ、GUP（プレビュー）を指定（後述）
- 永続的ストレージ
  - Azure Files 共有の直接マウント可（後述）
- Co-scheduled グループ
  - 複数コンテナー グループのスケジュール設定（後述）
- 仮想ネットワークのデプロイ
  - Azure 仮想ネットワークにコンテナー インスタンスをデプロイ可（後述）

---

![bg auto](https://live.staticflickr.com/65535/49894249437_b0e7c29933_h.jpg)

# **Azure Docker 統合**

- 従来から、az cli, Azure Posh, ARM で ACI を操作するのはあった
- 6/25から、`docker cli` を使って、ACIにデプロイできるようになったというのが今回の趣旨
- 本番環境構築なら、az cli なんのネイティブメソッドを使った方が制限が少ない
- 参照：
  - 公式ドキュメント [Deploying Docker containers on Azure](https://docs.docker.com/engine/context/aci-integration/)
  - GitHub Issue [Docker ACI Integration (Beta)](https://github.com/docker/aci-integration-beta)

# **ACI の CPU確認（１）**

値段が気になるところ 同等のVMと比較してみよう。CPUを確認のためログイン

```sh
docker login azure
docker context create aci mycontext
docker context use
```

適当な、imageを起動
```sh
docker run -p 80:80 nginx
```

# **ACI の CPU確認（２）**

`docker ps` でコンテナIDを確認し中に入る
```sh
docker ps
docker exec -it <container-id> /bin/bash
cat /proc/cpuinfo | grep 'model name' | head -1
model name      : Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
```

そこそこのCPUが使われてる。余った、Aシリーズとかでは無い、、、、:+1:

# **価格比較（概算）**

- ベースとなるVMの価格、[D2 V3](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/ubuntu-advantage-essential/), E5-2673 v4 2.3G East US2
  - 1 core 8 GB 4,300円/月
- [ACIを30日連続稼働](https://azure.microsoft.com/en-us/pricing/details/container-instances/)で計算
  - 3,300(1 core) + 2,900(8G) = 6,200円
### 概ね1.44倍
- 参考：[App Service](https://azure.microsoft.com/en-us/pricing/details/app-service/windows/) S1(1 core, 1.75GB)、8,176円、P1v2(1 core, 3.50GB)で16,352円

# **雑感（２）**

- 立ち上げっぱなしのサービスで使うには少し高い :-1:
  - マネージドサービスで管理は楽、秒課金はいい :+1:
- 起動は言うほど速くない :-1:
  - でもまあ結構速い :+1:
- 起動と停止がお手軽すぎる
  - 一瞬どこで動かしているか忘れる
- docker cliの補完が今風 CUI

# **マルチコンテナ アプリ（１）**

.NET Core で簡単な**あぷり**を書いたのでDeploy

- [docker-aci-integration-example](https://github.com/takekazuomi/docker-aci-integration-example)
- コンテナグループに、asp.net core と redis の２つのコンテナをデプロイする
- [docker-compose.yml](https://github.com/takekazuomi/docker-aci-integration-example/blob/master/docker-compose.yml)、[docker-compose.override.yml](https://github.com/takekazuomi/docker-aci-integration-example/blob/master/docker-compose.override.yml) に書く

# **マルチコンテナ アプリ（２）**

```sh
docker conetxt use defult
docker-compose -f ./docker-compose.yml -f ./docker-compose.override.yml build
docker-compose -f ./docker-compose.yml -f ./docker-compose.override.yml push
docker -c mycontext compose  -f ./docker-compose.yml -f ./docker-compose.override.yml up
```

動いているのを見る

```sh
docker -c mycontext ps
```

ポータルで確認

# **マルチコンテナ アプリ（３）**

消す

```sh
docker -c mycontext compose down
```

# **Marp**

今回のスライドは、[Marp]((https://github.com/marp-team/marp-cli#docker))を使わせてもらいました。Docker で動かします :+1:

```sh
docker run --rm --init -v ${PWD}/docs:/home/marp/app -e LANG=${LANG} -p 8081:8080 -p 37717:37717 marpteam/marp-cli -s .
```

今回のコンテンツ
- [docker-aci-intro](https://github.com/takekazuomi/docker-aci-intro)


# **Bookmarks**

- [Docker ACI Integration (Beta)](https://github.com/docker/aci-integration-beta)
- [Azure Container Instance](https://docs.microsoft.com/ja-jp/azure/container-instances/container-instances-overview)
- [Docker for Windows](https://docs.docker.com/docker-for-windows/)
- [Windows Subsystem for Linux WSL2](https://docs.microsoft.com/en-us/windows/wsl/)
- [marp-cli](https://github.com/marp-team/marp-cli#docker)

# 終

![bg auto](https://live.staticflickr.com/819/40713808344_00d29bb98c_h.jpg)

