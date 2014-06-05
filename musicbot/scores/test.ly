\version "2.16.2"
{

<<
    \new Voice = "melody" {
    ais'4.
        <<
        {
        \voiceOne
        g4 f4
        }
        \new Voice {
            \voiceTwo
            d16
            }
            >>
        \oneVoice
        e4
    a4
        <<
        {
        \voiceOne
        g f
        }
        \new Voice {
            \voiceTwo
            d2
            }
        >>
        \oneVoice
        e4
}
>> 
}
