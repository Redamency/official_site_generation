# official_site_generation

官网站点生成模型

根据输入的 `query` 生成对应的官网站点域名；如果无法确定，返回 `none`。

## 输入

- `query`：需要查找的实体/关键词（例如：公司、品牌、组织名称）

## 输出

- `response`：官网站点域名（例如：`www.google.com`），或 `none`

## 示例

输入：

```text
query="谷歌"
```

输出：

```text
response="www.google.com"
```

