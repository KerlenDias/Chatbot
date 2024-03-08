import json

print('Criando base de treinamento...')
intents = {"intents":
    [
        {"tag": "cumprimentar",
         "patterns": ["Oi", "Como vai vc", "Como vc ta", "Olá", "Bom dia", "Boa tarde", "Boa noite", "Tudo bem"],
         "responses": ["Oi", "Tudo beleza!", "Blz?", "Tudo tranquilo"]
         },
        {"tag": "despedir",
         "patterns": ["Até logo", "Até breve", "Até mais", "Tchau", "Xau", "Bye"],
         "responses": ["Tchau!", "Até mais", "[]'s"]
         },
        # TEM MAIS COISA AQUI...
        {"tag": "ofender",
         "patterns": ["Você é feio", "Não gosto de vc", "vc é chato"],
         "responses": ["Não ligo!", "E o kiko?"]
         }
    ]
}
print('Base de treinamento criada!')
