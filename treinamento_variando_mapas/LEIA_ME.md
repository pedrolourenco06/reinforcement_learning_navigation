# Revisão baseada nos seus scripts originais

Esta revisão substitui a proposta anterior de reescrever o agente. O arquivo
ackermann_env_re.py é cópia integral do ambiente enviado. As classes QNetwork,
ReplayBuffer e DQNAgent de labirinto_dqn_re.py foram mantidas iguais às enviadas.

Mantidos: 251 entradas, 8 ações (5 frente + 3 ré), observação global original,
recompensa original (inclusive exploração e custo de ré), cinemática, dimensão
do corpo, tolerância de 0,30 m, timeout de 500 passos, Double/Dueling DQN,
n_step=3, buffer=300000, gamma=0,99, lr=3e-4, batch=128, tau=0,005 e treino a cada
2 passos. O treinamento começa do zero, com epsilon 1→0,05 em 600 mil passos.

## Arquivos

- labirinto_dqn_re.py: treinamento original adaptado, gráficos e checkpoints.
- ackermann_env_re.py: ambiente original sem alterações.
- tarefas_multi.py: gerenciador de mapas, alvos, posições e janela Pygame.
- avaliar_multi.py: avaliação desta revisão, com janela animada por padrão.
- gerar_graficos_csv.py: cria gráficos de CSVs já existentes.
- compat_248/: avaliação animada dos modelos do pacote anterior de 248 entradas.
- mapas/: mesmos 12 mapas de treino, 4 de validação e 4 de teste fornecidos antes.

## Instalação e treinamento

Abra o terminal nesta pasta, usando o ambiente virtual com PyTorch/DirectML que
você já possui. Dependências adicionais:

    python -m pip install numpy scipy Pillow pygame gymnasium matplotlib pyparsing

Não reinstale torch por esse comando: preserve a versão do seu torch-directml.
Para CPU em um ambiente novo, instale PyTorch separadamente.

    python labirinto_dqn_re.py --dispositivo directml --passos 1500000

No primeiro episódio e a cada 50 episódios, uma janela mostra mapa, alvo, corpo,
orientação e trajetória do robô, usando o render original. A janela fecha nos
episódios intermediários. A cada 20 episódios, os PNGs são atualizados:

- results_re_multi/dqn_reward.png: recompensa e média móvel de 50.
- results_re_multi/dqn_success_rate.png: sucesso móvel e distância do currículo.
- results_re_multi/dqn_loss.png: loss e média móvel; antes das atualizações,
  mostra uma mensagem em vez de inventar valores.
- results_re_multi/dqn_validation.png: sucesso em validação, após a primeira
  avaliação (25 mil passos por padrão, ao término do episódio).

Os gráficos também são salvos no fim e em uma interrupção controlada. São PNGs,
não janelas interativas do matplotlib. Os CSVs e métricas ficam na mesma pasta.
A média móvel foi vetorizada sem alterar a fórmula, para não atrasar o treino.

Ajustes opcionais:

    python labirinto_dqn_re.py --dispositivo directml --render-a-cada 20 --graficos-a-cada 10 --velocidade 4

--sem-render desativa apenas a animação, não os gráficos.
--velocidade altera apenas a visualização; o dt físico continua 0,2 s.
Espaço pausa a animação; Esc/fechar a janela interrompe o treinamento.

## Sorteio

A cada episódio são sorteados mapa, alvo, posição e orientação inicial. Todos
os 12 mapas de treino são elegíveis desde o início, sem as fases de mapas da
reescrita anterior. Foi mantido o currículo de distância original: começa em
5 m e aumenta em 2,5 m quando a janela de 50 episódios alcança 70% de sucesso.
A tarefa permanece fixa durante o episódio e o mapa conhecido é zerado.

O gerador verifica área livre para o corpo e conectividade numa grade com margem
conservadora. Isso não garante que qualquer percurso seja executável por Ackermann
em 500 passos. A rede não recebe a conectividade nem o mapa completo.
O mapa completo é usado pelo simulador e pelo gerador de tarefas.

A subclasse em tarefas_multi.py vetoriza update_known_map, preservando exatamente
os valores e o info_gain do laço original. Não substitui a janela por raios.
O sensor, a recompensa e as observações têm as limitações já discutidas;
esta revisão não pretende corrigir todos esses aspectos ao mesmo tempo.

Os limites físicos de cada mapa vêm da resolução PNG e dos px_por_metro do
manifesto. O seu labirinto6.png antigo não foi incluído automaticamente.
Para adicioná-lo, copie para mapas/treino e acrescente uma entrada ao manifesto:

    {"arquivo":"treino/labirinto6.png","px_por_metro":100,"split":"treino","dificuldade":2}

Faça isso antes de iniciar um treino: a retomada verifica que mapas e código
não mudaram. Salve o manifesto como uma lista JSON válida.

## Avaliação animada desta revisão (251 entradas)

    python avaliar_multi.py --modelo results_re_multi/dqn_best.pt --dispositivo directml --episodios 30

Renderiza por padrão. Enter avança ao próximo episódio, Espaço pausa e Esc sai.
Para avanço automático:

    python avaliar_multi.py --modelo results_re_multi/dqn_best.pt --dispositivo directml --automatico --velocidade 4

Para um PNG específico, com alvos e inícios sorteados nele:

    python avaliar_multi.py --modelo results_re_multi/dqn_best.pt --mapa mapas/treino/circuito_referencia.png --px-por-metro 100

PNG das trajetórias (quando há renderização), CSVs por episódio e resumo.csv
ficam em results_re_multi/avaliacao. --sem-render permite avaliar sem janela.
Os mapas de teste não participam das atualizações nem da escolha do melhor modelo.

## Resultados do pacote anterior de 248 entradas

Esses pesos não são compatíveis com a revisão de 251 entradas. Preserve a pasta
results_multi antiga. Para avaliar os modelos que já treinou com animação:

    python compat_248/avaliar_multi.py --modelo CAMINHO/results_multi/melhor.pt --mapas mapas/manifesto.json --dispositivo directml --automatico

O ambiente antigo e a arquitetura necessária estão isolados em compat_248; não
misture com o ambiente desta revisão. Essa compatibilidade foi mantida apenas
para avaliar seus resultados existentes, não para iniciar outro treinamento.

Para recuperar gráficos dos CSVs já existentes:

    python gerar_graficos_csv.py --pasta CAMINHO/results_multi

Gera recompensa, sucesso e passos por episódio. A loss do pacote anterior não
pode ser reconstruída porque não foi registrada no CSV/checkpoint antigo.
Não confundir isso com a loss da revisão atual, que é registrada e plotada.

## Checkpoints e retomada

Checkpoints completos são salvos a cada avaliação e no fim, incluindo replay,
otimizador, redes, geradores aleatórios, currículo e histórico. Podem ultrapassar
600 MB e precisam de memória extra durante salvamento; não os envie ao Git.
A pasta de saída não pode conter arquivos ao iniciar do zero.

    python labirinto_dqn_re.py --dispositivo directml --retomar results_re_multi/dqn_checkpoint.pt --passos 2000000

--passos é a meta TOTAL. Só checkpoints desta revisão podem ser retomados.
O código termina o episódio antes de finalizar, podendo ultrapassar a meta.
Em Ctrl+C/fechamento durante um episódio, os pesos atuais são salvos em
 dqn_interrompido_pesos.pt, mas a retomada completa usa o checkpoint anterior.
Se interromper antes da primeira avaliação, ainda não haverá checkpoint completo.

## Validação realizada

Executados em CPU: treino curto com atualizações reais da rede, produção de
quatro gráficos, salvamento e retomada de checkpoint, avaliação animada via
Pygame com driver de vídeo virtual, exportação de trajetórias, compatibilidade
de avaliação 248 e recuperação de gráficos dos dois formatos de CSV.
Conferidos por comparação estrutural: QNetwork, ReplayBuffer e DQNAgent iguais
às classes originais; ambiente original byte a byte; update_known_map vetorizado
com os mesmos resultados do laço original. Inspecionados gráfico e captura renderizada.
Não houve treinamento longo, medição de convergência ou execução em DirectML.
