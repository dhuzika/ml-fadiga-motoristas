# Sistema de Detecção de Fadiga - Versão Menos Sensível

## Visão Geral

Este documento descreve o funcionamento do sistema de detecção de fadiga com melhorias implementadas para reduzir falsos positivos e proporcionar alertas mais confiáveis.

## Arquitetura do Sistema

O sistema utiliza um modelo XGBoost otimizado com Optuna (94% de acurácia) combinado com um sistema de pós-processamento inteligente que analisa múltiplas variáveis antes de emitir alertas.

### Componentes Principais

1. **Modelo XGBoost**: Realiza predições baseadas em features extraídas (EAR, MAR, PERCLOS, blink rate)
2. **Sistema de Suavização Temporal**: Analisa histórico de predições
3. **Validação PERCLOS**: Verifica indicadores físicos de fadiga
4. **Controle de Frequência**: Gerencia intervalo entre alertas

## Sistema de Suavização Temporal

### Implementação

```python
class SuavizadorTemporal:
    def __init__(self, janela=6, threshold_consenso=0.67):
        # Analisa últimas 6 predições
        # Requer 67% de consenso para alertar
```

### Funcionamento

- Mantém histórico das **6 últimas predições**
- Só alerta se **4 das 6** predições (67%) indicarem "Sonolento"
- Cada predição deve ter **95% ou mais de confiança**

### Exemplo de Análise

```
Predições: [Alerta, Sonolento(96%), Sonolento(97%), Alerta, Sonolento(98%), Sonolento(95%)]
Análise: 4/6 predições "Sonolento" com alta confiança = 67%
Resultado: ALERTA APROVADO

Predições: [Alerta, Sonolento(96%), Alerta, Sonolento(94%), Alerta, Sonolento(97%)]
Análise: 2/6 predições válidas = 33% (uma predição com 94% é descartada)
Resultado: SEM ALERTA
```

## Validação PERCLOS

### Conceito

PERCLOS (Percentage of Eyelid Closure) representa o percentual de tempo que os olhos permanecem fechados.

### Implementação

```python
if perclos_medio < 40.0:  # PERCLOS < 40%
    return False  # Não alerta
```

### Critério de Validação

- **PERCLOS > 40%**: Indica olhos fechados por tempo significativo (possível fadiga)
- **PERCLOS < 40%**: Indica piscar normal ou olhos abertos (descarta alerta)

## Configurações do Sistema

### Parâmetros Principais

```python
# Suavizador Temporal
JANELA_TEMPORAL = 6          # Últimas 6 predições analisadas
CONSENSO_MINIMO = 0.67       # 67% devem indicar "Sonolento"

# Thresholds de Confiança
CONFIANCA_MINIMA = 0.95      # 95% de confiança mínima

# Validação Física
PERCLOS_MINIMO = 40.0        # 40% mínimo de tempo com olhos fechados

# Controle de Alertas
INTERVALO_ALERTAS = 10.0     # 10 segundos entre alertas
```

### Algoritmo de Decisão

```
PARA CADA FRAME:
  1. Modelo XGBoost faz predição
  2. Calcula PERCLOS atual
  3. Adiciona ao histórico temporal

  SE buffer >= 3 predições:
    SE PERCLOS_médio >= 40%:
      SE predições_sonolento_95%+ >= 67%:
        SE tempo_desde_último_alerta >= 10s:
          EMITIR ALERTA
```

## Como Usar

### Execução

```bash
# No Jupyter Notebook teste_tempo_real_alerta.ipynb
1. Célula 1: Seleção do Modelo (escolha Optuna)
2. Células 2-5: Inicialização (MediaPipe, modelo, detector)
3. Célula 6: Sistema Melhorado (versão menos sensível)
```

### Requisitos

- **Modelo**: XGBoost já treinado (não requer retreinamento)
- **Acurácia**: 94% mantida
- **Pipeline**: Inalterado
- **Processamento**: Apenas pós-processamento melhorado

### Interface

A interface exibe:
- **Status da predição atual**
- **Informações do consenso temporal**
- **PERCLOS em tempo real**
- **Configurações ativas**
- **Métricas fisiológicas**

Exemplo de saída:
```
Predição: Sonolento
Confiança: 96%
Consenso: 67% | PERCLOS: 45% | Buffer: 6/6
PERCLOS: 45%
```

## Cenários de Funcionamento

### Piscar Normal

```
Frame 1: Sonolento(92%) → Sem alerta (confiança < 95%)
Frame 2: Alerta(88%) → Adiciona ao buffer
Frame 3: Sonolento(96%) → Adiciona ao buffer
Frames 4-6: Analisa consenso → PERCLOS 25% → Sem alerta
```

**Resultado**: Sistema reconhece piscar normal e não alerta.

### Fadiga Real

```
Frames 1-6: Múltiplas predições Sonolento(95%+)
PERCLOS médio: 50%
Consenso: 83% (5/6 predições válidas)
```

**Resultado**: Sistema detecta fadiga real e emite alerta.

## Troubleshooting

### Sistema não alertando

**Verificar:**
1. **Confiança**: Predições têm > 95%?
2. **PERCLOS**: Está > 40%?
3. **Consenso**: 4 das 6 predições são "Sonolento" válidas?
4. **Tempo**: Passou 10s desde último alerta?

### Sistema muito conservador

**Ajustar parâmetros para menos conservador:**
```python
suavizador = SuavizadorTemporal(janela=5, threshold_consenso=0.60)
PERCLOS_MINIMO = 30.0
CONFIANCA_MINIMA = 0.93
```

### Sistema pouco conservador

**Ajustar parâmetros para mais conservador:**
```python
suavizador = SuavizadorTemporal(janela=8, threshold_consenso=0.75)
PERCLOS_MINIMO = 50.0
CONFIANCA_MINIMA = 0.97
```

## Logs de Monitoramento

### Saída Durante Execução

```
FPS: 28.5 | Status: Sonolento | Conf: 0.96 | PERCLOS: 45.2% | Consenso: 67% | Buffer: 6/6
```

### Interpretação dos Logs

- **Status**: Predição atual do modelo
- **Conf**: Confiança da predição atual (0-1)
- **PERCLOS**: Percentual de tempo com olhos fechados
- **Consenso**: Percentual de predições "Sonolento" no buffer
- **Buffer**: Número de predições armazenadas

## Características do Sistema

### Principais Benefícios

- **Estabilidade**: Redução significativa de oscilações entre estados
- **Precisão**: Alertas baseados em múltiplas validações
- **Confiabilidade**: Elimina falsos positivos de piscar normal
- **Robustez**: Mantém detecção de fadiga real
- **Usabilidade**: Interface clara com informações de debug

### Especificações Técnicas

- **Threshold de confiança**: 95%
- **Janela temporal**: 6 predições
- **Consenso mínimo**: 67%
- **Validação PERCLOS**: 40% mínimo
- **Intervalo entre alertas**: 10 segundos
- **Taxa de processamento**: ~30 FPS

## Arquivos e Dependências

### Arquivos Modificados
- `teste_tempo_real_alerta.ipynb` → Célula 6 atualizada com sistema melhorado

### Arquivos Inalterados
- `treino_xgboost.ipynb` → Modelo original preservado
- `modelos_xgb_optuna/` → Modelos treinados mantidos
- `pipeline.py` → Pipeline de predição inalterado

### Dependências
- OpenCV, MediaPipe, NumPy (existentes)
- `collections.deque` (Python padrão)
- Nenhuma nova dependência adicionada

---

**Sistema**: Detecção de Fadiga Menos Sensível v1.0
**Plataforma**: Linux (alertas sonoros adaptados)
**Modelo**: XGBoost + Optuna (94% acurácia)
**Data**: 2025-10-12